"""NBDNet architecture (Kevala & Lal, 2024), adapted for the 9-channel
polarimetric representation this pipeline produces (cov_to_repr) instead
of the paper's single-channel SAR intensity.

Reconstructed from the paper's Fig. 1-4 and text description -- the paper
doesn't give numeric kernel sizes/layer counts beyond what's drawn, so
this is a best-effort faithful implementation of the described topology:
    Input -> noise estimation subnetwork -> estimated noise
    [input, estimated noise] concatenated -> non-blind denoising
    subnetwork (SE + CBAM attention, conv blocks 64/128/256/128/64/64,
    one skip connection around the 256-channel bottleneck) -> output

Trained with MSE loss between clean (C_ref) and noisy (C_noisy)
9-channel representations, per the paper's training setup.
"""
import keras
from keras import layers


@keras.saving.register_keras_serializable(package="nbdnet")
class ChannelMeanPool(layers.Layer):
    def call(self, x):
        return keras.ops.mean(x, axis=-1, keepdims=True)


@keras.saving.register_keras_serializable(package="nbdnet")
class ChannelMaxPool(layers.Layer):
    def call(self, x):
        return keras.ops.max(x, axis=-1, keepdims=True)


def squeeze_excitation_block(x, reduction=8, name=None):
    # fc1/fc2 have no BatchNormalization around them (unlike conv_block),
    # so under mixed_float16 they're unguarded fp16 compute, same risk as
    # estimated_noise/output_image in build_nbdnet -- pin to float32.
    channels = x.shape[-1]
    se = layers.GlobalAveragePooling2D(keepdims=True, name=f"{name}_gap" if name else None)(x)
    se = layers.Conv2D(max(channels // reduction, 4), 1, activation="relu",
                        dtype="float32", name=f"{name}_fc1" if name else None)(se)
    se = layers.Conv2D(channels, 1, activation="sigmoid",
                        dtype="float32", name=f"{name}_fc2" if name else None)(se)
    return layers.Multiply(name=f"{name}_scale" if name else None)([x, se])


def channel_attention(x, reduction=8, name=None):
    channels = x.shape[-1]
    avg = layers.GlobalAveragePooling2D(keepdims=True)(x)
    mx = layers.GlobalMaxPooling2D(keepdims=True)(x)

    # Same reasoning as squeeze_excitation_block: fc1/fc2 have no BN, and
    # fc2's output is unbounded (no activation) before the Add+sigmoid below.
    fc1 = layers.Conv2D(max(channels // reduction, 4), 1, activation="relu",
                         dtype="float32", name=f"{name}_ca_fc1" if name else None)
    fc2 = layers.Conv2D(channels, 1, dtype="float32", name=f"{name}_ca_fc2" if name else None)

    avg_out = fc2(fc1(avg))
    max_out = fc2(fc1(mx))
    attn = layers.Activation("sigmoid")(layers.Add()([avg_out, max_out]))
    return layers.Multiply(name=f"{name}_ca_scale" if name else None)([x, attn])


def spatial_attention(x, kernel_size=7, name=None):
    avg = ChannelMeanPool(name=f"{name}_avgpool" if name else None)(x)
    mx = ChannelMaxPool(name=f"{name}_maxpool" if name else None)(x)
    concat = layers.Concatenate(axis=-1)([avg, mx])
    # Same reasoning as above -- sa_conv has no BN either.
    attn = layers.Conv2D(1, kernel_size, padding="same", activation="sigmoid", dtype="float32",
                          name=f"{name}_sa_conv" if name else None)(concat)
    return layers.Multiply(name=f"{name}_sa_scale" if name else None)([x, attn])


def cbam_block(x, reduction=8, kernel_size=7, name=None):
    x = channel_attention(x, reduction=reduction, name=name)
    x = spatial_attention(x, kernel_size=kernel_size, name=name)
    return x


def conv_block(x, filters, n_layers=1, kernel_size=3, name=None):
    for i in range(n_layers):
        x = layers.Conv2D(filters, kernel_size, padding="same", name=f"{name}_conv{i}" if name else None)(x)
        x = layers.BatchNormalization(name=f"{name}_bn{i}" if name else None)(x)
        x = layers.ReLU(name=f"{name}_relu{i}" if name else None)(x)
    return x


def noise_estimation_subnetwork(inputs, channels, filters=32, n_layers=3):
    x = conv_block(inputs, filters, n_layers=n_layers, name="noise_est")
    # No BatchNorm follows this conv (unlike every other block), so under
    # mixed_float16 it's the first unguarded fp16 compute in the graph --
    # force float32 here to stop it overflowing before the concat below.
    noise = layers.Conv2D(channels, 3, padding="same", name="estimated_noise", dtype="float32")(x)
    return noise


def non_blind_denoising_subnetwork(x, out_channels):
    x = conv_block(x, 64, n_layers=1, name="nbd_64a")
    x = squeeze_excitation_block(x, name="nbd_64a")
    x = cbam_block(x, name="nbd_64a")

    skip = conv_block(x, 128, n_layers=1, name="nbd_128")

    x = conv_block(skip, 256, n_layers=2, name="nbd_256")

    x = layers.Conv2D(128, 3, padding="same", name="nbd_bottleneck_proj")(x)
    x = layers.BatchNormalization(name="nbd_bottleneck_proj_bn")(x)
    x = layers.Add(name="nbd_skip_add")([x, skip])
    x = squeeze_excitation_block(x, name="nbd_128b")
    x = cbam_block(x, name="nbd_128b")

    x = conv_block(x, 64, n_layers=1, name="nbd_64b")
    x = squeeze_excitation_block(x, name="nbd_64b")
    x = cbam_block(x, name="nbd_64b")

    x = conv_block(x, 64, n_layers=1, name="nbd_64c")
    # Same reasoning as estimated_noise above, plus this feeds MSE loss
    # directly -- TF mixed-precision guidance is to keep the model's final
    # output float32 regardless.
    out = layers.Conv2D(out_channels, 3, padding="same", activation=None, name="output_image", dtype="float32")(x)
    return out


def build_nbdnet(input_shape=(64, 64, 9)):
    """input_shape: (H, W, C) -- C=9 for the cov_to_repr representation
    this pipeline uses (3 normalized-dB powers + 6 coherence real/imag).
    """
    channels = input_shape[-1]
    inputs = keras.Input(shape=input_shape, name="noisy_input")

    estimated_noise = noise_estimation_subnetwork(inputs, channels)
    merged = layers.Concatenate(axis=-1, name="merge_input_noise")([inputs, estimated_noise])

    output = non_blind_denoising_subnetwork(merged, out_channels=channels)
    # residual: network predicts the clean signal directly (paper trains
    # against clean/noisy pairs with MSE, not a noise residual)
    model = keras.Model(inputs=inputs, outputs=output, name="NBDNet")
    return model


if __name__ == "__main__":
    m = build_nbdnet((64, 64, 9))
    m.summary()
