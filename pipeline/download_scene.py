"""Download UAVSAR products for a scene from ASF (Alaska Satellite Facility).

Requires:
    pip install asf_search
    a free NASA Earthdata account: https://urs.earthdata.nasa.gov/users/new

Usage:
    python -m pipeline.download_scene housto_32915_17087_005_170831_L090_CX_01 \
        --out-dir pipeline/data/raw/housto_32915_17087_005_170831_L090_CX_01

This queries ASF's live catalog for whatever product files actually exist
for the given scene ID and downloads them. It does NOT guess file URLs --
everything it downloads comes from a real search result.

Important: ASF's UAVSAR holdings are mostly MLC/GRD-level (multilooked,
sometimes geocoded) plus DEM/incidence/slope layers. Verified check
against 500 sampled UAVSAR granules on ASF found zero with a raw SLC
(.slc) file attached -- SLC-level UAVSAR data does not appear to be
distributed through ASF's standard catalog. If this script reports "no
SLC found" for your scene, that matches this pattern; the PreProc_Guide
pipeline needs SLC specifically (Step 1), so you'd either need to:
  - request true SLC access directly from the JPL UAVSAR team
    (contact listed at https://uavsar.jpl.nasa.gov), or
  - adapt the preprocessing pipeline to start from MLC instead (skip the
    custom multilook in covariance.py and the sub-aperture route in
    subaperture.py, since MLC is already multilooked once).
"""
import argparse
import getpass
import os

import asf_search as asf

ALL_EXTENSIONS = ("slc", "mlc", "grd", "dat", "hgt", "inc", "slope", "ann", "kmz")


def find_scene(scene_id):
    results = asf.search(platform=asf.PLATFORM.UAVSAR, granule_list=[scene_id])
    if not results:
        raise SystemExit(f"no ASF granule found for scene_id={scene_id!r}")
    return results[0]


def list_available_products(granule):
    """Return {extension: [(filename, url, bytes), ...]} from a granule's
    bytes dict + additionalUrls, matched up by filename.
    """
    byte_info = granule.properties.get("bytes") or {}
    urls_by_name = {}
    for u in granule.properties.get("additionalUrls") or []:
        urls_by_name[os.path.basename(u)] = u
    main_name = granule.properties.get("fileName")
    if main_name:
        urls_by_name[main_name] = granule.properties.get("url")

    by_ext = {}
    for filename, info in byte_info.items():
        ext = filename.rsplit(".", 1)[-1].lower()
        url = urls_by_name.get(filename)
        by_ext.setdefault(ext, []).append((filename, url, info.get("bytes")))
    return by_ext


def download_scene(scene_id, out_dir, extensions, username=None, password=None, processes=1):
    granule = find_scene(scene_id)
    by_ext = list_available_products(granule)

    print(f"scene {scene_id}: available product types = {sorted(by_ext)}")
    if "slc" not in by_ext:
        print(
            "warning: no raw SLC (.slc) files found for this scene on ASF. "
            "See module docstring -- SLC access likely needs a direct JPL request."
        )

    to_fetch = []
    for ext in extensions:
        for filename, url, nbytes in by_ext.get(ext, []):
            if url is None:
                print(f"skipping {filename}: no URL in search result")
                continue
            to_fetch.append((filename, url, nbytes))

    if not to_fetch:
        raise SystemExit(f"none of the requested extensions {extensions} are available for this scene")

    total_bytes = sum(n or 0 for _, _, n in to_fetch)
    print(f"downloading {len(to_fetch)} files, {total_bytes / 1e9:.2f} GB total, to {out_dir}")
    for filename, url, nbytes in to_fetch:
        print(f"  {filename}  ({(nbytes or 0) / 1e9:.2f} GB)")

    os.makedirs(out_dir, exist_ok=True)

    session = asf.ASFSession()
    if not (username and password):
        username = username or input("NASA Earthdata username: ")
        password = password or getpass.getpass("NASA Earthdata password: ")
    session.auth_with_creds(username, password)

    asf.download_urls(urls=[url for _, url, _ in to_fetch], path=out_dir, session=session, processes=processes)
    print("done.")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("scene_id", help='UAVSAR scene id, e.g. "housto_32915_17087_005_170831_L090_CX_01"')
    p.add_argument("--out-dir", default=None, help="defaults to pipeline/data/raw/<scene_id>")
    p.add_argument(
        "--extensions",
        nargs="+",
        default=["slc", "ann"],
        choices=ALL_EXTENSIONS,
        help="which product types to download (default: slc + ann, per the preprocessing guide)",
    )
    p.add_argument("--username", default=os.environ.get("EARTHDATA_USERNAME"))
    p.add_argument("--password", default=os.environ.get("EARTHDATA_PASSWORD"))
    p.add_argument("--processes", type=int, default=1)
    p.add_argument("--list-only", action="store_true", help="only report available product types, don't download")
    args = p.parse_args()

    out_dir = args.out_dir or os.path.join("pipeline", "data", "raw", args.scene_id)

    if args.list_only:
        granule = find_scene(args.scene_id)
        by_ext = list_available_products(granule)
        for ext, files in sorted(by_ext.items()):
            total = sum(n or 0 for _, _, n in files) / 1e9
            print(f"{ext}: {len(files)} file(s), {total:.2f} GB")
        return

    download_scene(args.scene_id, out_dir, args.extensions, args.username, args.password, args.processes)


if __name__ == "__main__":
    main()
