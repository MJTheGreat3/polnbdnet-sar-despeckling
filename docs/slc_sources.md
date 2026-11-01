# UAVSAR SLC source candidates

Candidate UAVSAR "Coregistered SLC stack" product pages, surveyed for terrain
diversity beyond the current BERMS (boreal wetland/forest, Saskatchewan) and
`monda` campaigns. All 14 below are confirmed quad-pol (HH/HV/VH/VV) and
confirmed stack-SLC products (raw `.slc` present) — not InSAR Pair/RPI
products, which only ship multilooked derived products (interferogram,
coherence, amplitude) and have **no raw SLC at all**.

All 14 real `.ann` URLs below were extracted directly from the printed product
pages (`docs/sources/*.pdf`) with `pdftotext` — not guessed. Each resolves
`200` via `curl -I`. The 4 matching `.slc` files (one per pol) sit next to
each `.ann` with the same filename stem, suffixed `_s1_1x1.slc` instead of
`.ann`.

| Site | Terrain/location | Quad-pol | Size/tracks | Verdict | Verified `.ann` URL (HH, first track) |
|---|---|---|---|---|---|
| `rockmt_14107` | Rocky Mountains NP, CO — alpine | y | many tracks, ~5.1GB/pol/track | **use** | `downloaduav2.jpl.nasa.gov/Release32/rockmt_14107_01/rockmt_14107_20005_009_200212_L090HH_01_BU.ann` |
| `cvalle_32524` | Central Valley, CA — agricultural | y | 3 segs, 9 tracks, ~4.8GB/pol | **use** | `downloaduav2.jpl.nasa.gov/Release33/cvalle_32524_01/cvalle_32524_23007_005_230629_L090HH_01_BC.ann` |
| `barrow_15018` | Barrow Legacy Line, AK — Arctic coastal tundra | y | 3 segs, ~5.0GB/pol | **use** | `downloaduav2.jpl.nasa.gov/Release34/barrow_15018_01/barrow_15018_17069_003_170621_L090HH_01_BC.ann` |
| `atchaf_06309` | Atchafalaya River Delta, LA — subtropical wetland | y | many tracks, ~5.1GB/pol/track | **use** | `downloaduav2.jpl.nasa.gov/Release27/atchaf_06309_02/atchaf_06309_21023_001_210327_L090HH_02_BU.ann` |
| `dorado_34017` | Eldorado National Forest, CA — Sierra Nevada montane forest | y | 3 segs, ~4.7GB/pol | **use** | `downloaduav2.jpl.nasa.gov/Release28/dorado_34017_01/dorado_34017_20002_002_200131_L090HH_01_BU.ann` |
| `stlake_27129` | Salt Lake City, UT — urban/mountain valley | y | 1 seg, ~3.8GB/pol | **use** | `downloaduav2.jpl.nasa.gov/Release31/stlake_27129_01/stlake_27129_20007_004_200213_L090HH_01_BC.ann` |
| `trinit_14940` | Trinity River, TX — river/Gulf coastal plain | y | 3 segs, ~4.9GB/pol | maybe — unique terrain, lower priority | `downloaduav2.jpl.nasa.gov/Release28/trinit_14940_01/trinit_14940_17087_010_170831_L090HH_01_BC.ann` |
| `uticam_21003` | Utica, MT — plains/prairie | y | 2 segs, ~5.0GB/pol | maybe — unique terrain, lower priority | `downloaduav2.jpl.nasa.gov/Release20/uticam_21003_01/uticam_21003_21002_003_210115_L090HH_01_BC.ann` |
| `aklavi_25702` | Aklavik Highway, NWT Canada — Arctic tundra (ABoVE) | y | 3 segs, ~5.0GB/pol | skip — redundant with `barrow`, same terrain bucket | `downloaduav2.jpl.nasa.gov/Release31/aklavi_25702_01/aklavi_25702_17070_003_170622_L090HH_01_BC.ann` |
| `anaktW_16405` | Anaktuvuk Burn Scar West, AK — Arctic tundra burn scar | y | 3 segs, ~5.0GB/pol | skip — redundant with `barrow`; only useful if fire-disturbed tundra specifically wanted | `downloaduav2.jpl.nasa.gov/Release24/anaktW_16405_01/anaktW_16405_17069_005_170621_L090HH_01_BC.ann` |
| `atchaf_19809` | Atchafalaya Delta, LA (same site, different track) | y | many tracks, ~4.6GB/pol/track | skip — duplicate terrain of `atchaf_06309` | `downloaduav2.jpl.nasa.gov/Release30/atchaf_19809_02/atchaf_19809_21023_000_210327_L090HH_02_BU.ann` |
| `wterre_16300` | West Terrebonne Basin, LA — coastal wetland/bayou | y | 2 segs, many tracks, ~5.0GB/pol | skip — redundant, same LA wetland bucket | `downloaduav2.jpl.nasa.gov/Release31/wterre_16300_02/wterre_16300_21026_000_210405_L090HH_02_BU.ann` |
| `NISARP_20706` | Yucatan Lake, LA — lake/wetland | y | 3 segs, ~5.0GB/pol | skip — redundant, same LA wetland bucket | `downloaduav2.jpl.nasa.gov/Release31/NISARP_20706_01/NISARP_20706_19040_011_190622_L090HH_01_BU.ann` |
| `alamos_35915` | Los Alamos, NM — montane forest/sagebrush | y | 1 seg, ~5.1GB/pol | skip — redundant with `dorado`, same forest bucket, smaller | `downloaduav2.jpl.nasa.gov/Release20/alamos_35915_04/alamos_35915_20005_003_200212_L090HH_04_BU.ann` |

## Decision

Selected the 6 **use** sites — 6 distinct terrain types (alpine, agricultural,
Arctic coastal tundra, subtropical wetland, montane forest, urban/mountain
valley), zero overlap with each other or with BERMS/`monda`. `trinit`/`uticam`
are optional 7th/8th picks if more diversity is wanted later. The 6 `skip`
sites were dropped only because a stronger pick already covers the same
terrain bucket, not because of any data-quality issue.

Plan (report.md §3.3): multi-crop each selected site via the native
flat-binary `.ann`+`.slc` legacy format (no `nisar_to_legacy.py` HDF5
conversion needed — these are UAVSAR legacy stack products already in that
format), fold into train, hold out at least one track per site as an
additional test scene.

## Raw candidate list (source URLs)

```
https://uavsar.jpl.nasa.gov/cgi-bin/product.pl?jobName=NISARP_20706_01#data
https://uavsar.jpl.nasa.gov/cgi-bin/product.pl?jobName=stlake_27129_01#data
https://uavsar.jpl.nasa.gov/cgi-bin/product.pl?jobName=trinit_14940_01#data
https://uavsar.jpl.nasa.gov/cgi-bin/product.pl?jobName=wterre_16300_02#data
https://uavsar.jpl.nasa.gov/cgi-bin/product.pl?jobName=uticam_21003_01#data
https://uavsar.jpl.nasa.gov/cgi-bin/product.pl?jobName=anaktW_16405_01#data
https://uavsar.jpl.nasa.gov/cgi-bin/product.pl?jobName=atchaf_06309_02#data
https://uavsar.jpl.nasa.gov/cgi-bin/product.pl?jobName=atchaf_19809_02#data
https://uavsar.jpl.nasa.gov/cgi-bin/product.pl?jobName=barrow_15018_01#data
https://uavsar.jpl.nasa.gov/cgi-bin/product.pl?jobName=cvalle_32524_01#data
https://uavsar.jpl.nasa.gov/cgi-bin/product.pl?jobName=dorado_34017_01#data
https://uavsar.jpl.nasa.gov/cgi-bin/product.pl?jobName=alamos_35915_04#data
https://uavsar.jpl.nasa.gov/cgi-bin/product.pl?jobName=aklavi_25702_01#data
https://uavsar.jpl.nasa.gov/cgi-bin/product.pl?jobName=rockmt_14107_01#data
```
