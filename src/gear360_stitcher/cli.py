"""CLI: stitch a Samsung Gear 360 dual-fisheye photo or video into an
equirectangular panorama.

Video defaults differ from photo defaults (see VideoStitchConfig) and are
tuned for speed at full resolution; the flags below buy quality back.

Examples:
    gear360-stitch 360_0439.JPG -o 360_0439_equirect.jpg
    gear360-stitch 360_0439.MP4 -o 360_0439_equirect.mp4
    gear360-stitch 360_0439.MP4 -o slow_but_pretty.mp4 \\
        --anti-alias adaptive --seam-algorithm graphcut --seam-interval 1
"""

import argparse
import dataclasses
import logging
import os
import struct
import sys
from pathlib import Path

import cv2
import numpy as np

from .stitching import (
    FisheyeCamera,
    Gear360Stitcher,
    StitchConfig,
    VideoStitchConfig,
    estimate_level_rotation,
    estimate_level_rotation_from_video,
    load_image,
    patch_directions,
    render_center_patch,
    save_image,
    split_dual_fisheye,
    stitch_video_parallel,
)

logger = logging.getLogger(__name__)

# Bundled default calibration (Gear 360 unit calibrated with autocalib).
DATA_DIR = Path(__file__).resolve().parent / "data"
DEFAULT_CALIB_FRONT = DATA_DIR / "front"
DEFAULT_CALIB_BACK = DATA_DIR / "back"
DEFAULT_EXTRINSICS = DATA_DIR / "R_t.txt"

VIDEO_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv", ".m4v", ".webm"}


_XMP_HEADER = b"http://ns.adobe.com/xap/1.0/\x00"


def _gpano_xmp(width: int, height: int) -> bytes:
    props = {
        "ProjectionType": "equirectangular",
        "UsePanoramaViewer": "True",
        "FullPanoWidthPixels": width,
        "FullPanoHeightPixels": height,
        "CroppedAreaImageWidthPixels": width,
        "CroppedAreaImageHeightPixels": height,
        "CroppedAreaLeftPixels": 0,
        "CroppedAreaTopPixels": 0,
    }
    attrs = "\n      ".join(f'GPano:{k}="{v}"' for k, v in props.items())
    return (
        '<?xpacket begin="\ufeff" id="W5M0MpCehiHzreSzNTczkc9d"?>\n'
        '<x:xmpmeta xmlns:x="adobe:ns:meta/">\n'
        ' <rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">\n'
        '  <rdf:Description rdf:about=""\n'
        '      xmlns:GPano="http://ns.google.com/photos/1.0/panorama/"\n'
        f"      {attrs}/>\n"
        " </rdf:RDF>\n"
        "</x:xmpmeta>\n"
        '<?xpacket end="w"?>'
    ).encode("utf-8")


def tag_equirectangular(path: Path, width: int, height: int) -> None:
    """Embed the Google photo-sphere (XMP-GPano) metadata so viewers such as
    Google Photos treat the image as a 360 panorama, by inserting an XMP APP1
    segment into the JPEG (no exiftool needed). Those viewers also only accept
    JPEG, so any other format is left untouched with a warning."""
    data = path.read_bytes()
    if not data.startswith(b"\xff\xd8"):
        logger.warning("%s is not a JPEG; Google Photos won't show it as spherical", path)
        return
    payload = _XMP_HEADER + _gpano_xmp(width, height)
    segment = b"\xff\xe1" + (len(payload) + 2).to_bytes(2, "big") + payload
    # Insert after SOI and an optional JFIF APP0 segment, as the spec orders them.
    pos = 2
    if data[pos:pos + 2] == b"\xff\xe0":
        pos += 2 + int.from_bytes(data[pos + 2:pos + 4], "big")
    path.write_bytes(data[:pos] + segment + data[pos:])


_MP4_EXTENSIONS = {".mp4", ".mov", ".m4v"}
_SPHERICAL_V1_UUID = bytes.fromhex("ffcc8263f8554a938814587a02521fdd")
_SPHERICAL_V1_XML = (
    '<?xml version="1.0"?><rdf:SphericalVideo '
    'xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#" '
    'xmlns:GSpherical="http://ns.google.com/videos/1.0/spherical/">'
    "<GSpherical:Spherical>true</GSpherical:Spherical>"
    "<GSpherical:Stitched>true</GSpherical:Stitched>"
    "<GSpherical:StitchingSoftware>gear360-stitcher</GSpherical:StitchingSoftware>"
    "<GSpherical:ProjectionType>equirectangular</GSpherical:ProjectionType>"
    "</rdf:SphericalVideo>"
).encode("utf-8")
# Optional boxes that end a VisualSampleEntry; the V2 spec wants sv3d before them.
_OPTIONAL_VISUAL_BOXES = {b"clap", b"pasp", b"colr", b"btrt", b"fiel", b"m4ds", b"chrm", b"gama"}
# VisualSampleEntry fields between its box header and its child boxes.
_VISUAL_ENTRY_FIELDS = 78
# Boxes whose children are other boxes, on the path to the video sample entry
# and the chunk-offset tables.
_MP4_CONTAINERS = {b"moov", b"trak", b"mdia", b"minf", b"stbl"}


def _box(kind: bytes, payload: bytes) -> bytes:
    return struct.pack(">I", 8 + len(payload)) + kind + payload


def _spherical_v2_box() -> bytes:
    """sv3d (svhd + proj(prhd + equi)): mono equirectangular, full frame."""
    svhd = _box(b"svhd", b"\0\0\0\0" + b"gear360-stitcher\0")
    prhd = _box(b"prhd", struct.pack(">Iiii", 0, 0, 0, 0))
    equi = _box(b"equi", struct.pack(">IIIII", 0, 0, 0, 0, 0))
    return _box(b"sv3d", svhd + _box(b"proj", prhd + equi))


def _mp4_children(buf: bytes, start: int, end: int):
    """(type, box_start, box_end) of each 32-bit-sized box in buf[start:end]."""
    pos = start
    while pos + 8 <= end:
        size, kind = struct.unpack_from(">I4s", buf, pos)
        if size < 8 or pos + size > end:
            raise ValueError(f"unsupported or corrupt MP4 box {kind!r} at {pos}")
        yield kind, pos, pos + size
        pos += size


def _is_video_trak(moov: bytes, trak: tuple) -> bool:
    for kind, s, e in _mp4_children(moov, trak[1] + 8, trak[2]):
        if kind == b"mdia":
            for k, cs, ce in _mp4_children(moov, s + 8, e):
                if k == b"hdlr":
                    return moov[cs + 16:cs + 20] == b"vide"
    return False


def _shift_chunk_offsets(moov: bytearray, start: int, end: int, delta: int) -> None:
    for kind, s, e in list(_mp4_children(moov, start, end)):
        if kind in _MP4_CONTAINERS:
            _shift_chunk_offsets(moov, s + 8, e, delta)
        elif kind in (b"stco", b"co64"):
            fmt, width = (">I", 4) if kind == b"stco" else (">Q", 8)
            (count,) = struct.unpack_from(">I", moov, s + 12)
            for i in range(count):
                off = s + 16 + i * width
                (val,) = struct.unpack_from(fmt, moov, off)
                struct.pack_into(fmt, moov, off, val + delta)


def tag_equirectangular_video(path: Path, v1: bool = True, v2: bool = True) -> None:
    """Mark an MP4/MOV as a 360 equirectangular video for YouTube/Google Photos
    by adding the spherical-video metadata (V2 sv3d box in the video sample
    entry, V1 uuid XML box in the trak) to the container. Only the moov box is
    rewritten; the media data is copied verbatim."""
    if path.suffix.lower() not in _MP4_EXTENSIONS:
        logger.warning("%s is not an MP4/MOV; no spherical metadata written", path)
        return
    with open(path, "rb") as f:
        pos, moov_at, mdat_at, size_total = 0, None, None, os.fstat(f.fileno()).st_size
        while pos + 8 <= size_total:
            f.seek(pos)
            size, kind = struct.unpack(">I4s", f.read(8))
            if size == 1:
                size = struct.unpack(">Q", f.read(8))[0]
            elif size == 0:
                size = size_total - pos
            if kind in (b"moof", b"mvex"):
                raise ValueError("fragmented MP4 is not supported")
            if kind == b"moov":
                moov_at, moov_size = pos, size
            elif kind == b"mdat" and mdat_at is None:
                mdat_at = pos
            pos += size
        if moov_at is None:
            raise ValueError(f"{path} has no moov box")
        f.seek(moov_at)
        moov = bytearray(f.read(moov_size))

    traks = [c for c in _mp4_children(moov, 8, len(moov)) if c[0] == b"trak"]
    trak = next((t for t in traks if _is_video_trak(moov, t)), None)
    if trak is None:
        raise ValueError(f"{path} has no video track")
    chain = [trak]  # trak -> mdia -> minf -> stbl -> stsd
    for want in (b"mdia", b"minf", b"stbl", b"stsd"):
        parent = chain[-1]
        # stsd's children start after its version/flags + entry_count
        child = next(c for c in _mp4_children(moov, parent[1] + 8, parent[2]) if c[0] == want)
        chain.append(child)
    stsd = chain[-1]
    entry = next(iter(_mp4_children(moov, stsd[1] + 16, stsd[2])))
    if b"sv3d" in moov[entry[1]:entry[2]]:
        logger.info("%s already has spherical metadata", path)
        return

    v2 = _spherical_v2_box() if v2 else b""
    v1 = _box(b"uuid", _SPHERICAL_V1_UUID + _SPHERICAL_V1_XML) if v1 else b""
    delta = len(v1) + len(v2)

    # Chunk offsets are absolute: growing moov moves mdat only if it follows moov.
    if mdat_at is not None and mdat_at > moov_at:
        _shift_chunk_offsets(moov, 8, len(moov), delta)
    # sv3d goes after the codec config but before the optional trailing boxes.
    sv3d_at = next(
        (cs for k, cs, _ in _mp4_children(moov, entry[1] + 8 + _VISUAL_ENTRY_FIELDS, entry[2])
         if k in _OPTIONAL_VISUAL_BOXES),
        entry[2],
    )
    moov[trak[2]:trak[2]] = v1       # later position first, so earlier ones stay valid
    moov[sv3d_at:sv3d_at] = v2
    # moov and trak gained both boxes; mdia..stsd and the entry only the sv3d.
    grown = [(0, delta), (trak[1], delta)]
    grown += [(box[1], len(v2)) for box in (*chain[1:], entry)]
    for box_start, grow in grown:
        (size,) = struct.unpack_from(">I", moov, box_start)
        struct.pack_into(">I", moov, box_start, size + grow)

    tmp = path.with_name(path.name + ".tmp")
    with open(path, "rb") as src, open(tmp, "wb") as dst:
        _copy_bytes(src, dst, moov_at)
        dst.write(moov)
        src.seek(moov_at + moov_size)
        _copy_bytes(src, dst, None)
    tmp.replace(path)


def _copy_bytes(src, dst, n) -> None:
    while n is None or n > 0:
        chunk = src.read(1 << 20 if n is None else min(n, 1 << 20))
        if not chunk:
            break
        dst.write(chunk)
        if n is not None:
            n -= len(chunk)


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "input", type=Path,
        help="path to the dual-fisheye photo or video; video is detected by extension "
        f"({', '.join(sorted(VIDEO_EXTENSIONS))})",
    )
    parser.add_argument(
        "-o", "--output", type=Path, default=None,
        help="output path (default: <input>_equirect.<ext> next to the input, "
        "same container/image format)",
    )
    parser.add_argument(
        "--calib-front", type=Path, default=DEFAULT_CALIB_FRONT,
        help="directory with the front lens' calibration.json / vignette_*.json (default: bundled)",
    )
    parser.add_argument(
        "--calib-back", type=Path, default=DEFAULT_CALIB_BACK,
        help="directory with the back lens' calibration.json / vignette_*.json (default: bundled)",
    )
    parser.add_argument(
        "--extrinsics", type=Path, default=DEFAULT_EXTRINSICS,
        help="R_t.txt: back-camera-relative-to-front-camera 3x4 pose (default: bundled)",
    )
    parser.add_argument(
        "--mask-mode", choices=["seam", "simple"], default="seam",
        help="how to choose which lens covers each pixel near the two seams: "
        "'seam' finds a minimum-cost cut, 'simple' cuts at a fixed column (default: seam)",
    )
    parser.add_argument(
        "--anti-alias", choices=["off", "adaptive"], default=None,
        help="pre-filter each lens before remapping it, so downsampling from the "
        "calibrated resolution (much denser, angularly, than the panorama needs — "
        "see StitchConfig.output_width) gets properly blurred away instead of "
        "aliasing into visible noise. 'adaptive' sizes the blur per output pixel: "
        "none at all at each lens' optical axis, the most near the seams, where the "
        "fisheye is downsampled up to ~2x. Default: 'adaptive' for a photo, 'off' "
        "for video, where it roughly doubles the per-frame cost — pass it explicitly "
        "for video if you would rather have the quality, especially for a moving "
        "camera, where the aliasing it removes shimmers along the seams. (A third "
        "mode, 'fixed', applied one global blur and was removed — at these output "
        "widths it reduced to 'off' exactly)",
    )
    parser.add_argument(
        "--post-process", dest="post_process", action="store_true", default=None,
        help="denoise, contrast-boost, and sharpen the finished panorama (see "
        "enhance_panorama). Runs once on the already-blended image, independent of "
        "--anti-alias/--mask-mode/color-matching (default: on for a photo, off for "
        "video, where the extra bilateral-filter/CLAHE/sharpen passes add up per frame)",
    )
    parser.add_argument(
        "--no-post-process", dest="post_process", action="store_false",
        help="disable --post-process",
    )
    parser.add_argument(
        "--post-process-denoise", type=float, default=StitchConfig.post_process_denoise,
        help="bilateral-filter sigmaColor for the post-process denoise stage, "
        "0-255 scale, <= 0 to skip it (default: %(default)s)",
    )
    parser.add_argument(
        "--post-process-contrast", type=float, default=StitchConfig.post_process_contrast,
        help="CLAHE clip limit for the post-process local-contrast stage (on L* only), "
        "<= 0 to skip it (default: %(default)s)",
    )
    parser.add_argument(
        "--post-process-sharpen", type=float, default=StitchConfig.post_process_sharpen,
        help="unsharp-mask amount for the post-process sharpen stage, <= 0 to skip it "
        "(default: %(default)s)",
    )
    parser.add_argument(
        "--seam-algorithm", choices=["graphcut", "dp"], default=None,
        help="seam-finding algorithm when --mask-mode=seam: 'graphcut' is a true 2D "
        "min-cut, 'dp' a ~3x cheaper per-row dynamic program (default: "
        f"{StitchConfig.seam_algorithm} for a photo, {VideoStitchConfig.seam_algorithm} "
        "for a video, where the search runs per frame and seam_smoothing attenuates "
        "the wandering row-paths 'dp' can produce — see VideoStitchConfig)",
    )
    parser.add_argument(
        "--seam-interval", type=int, default=None,
        help="recompute the seam from scratch every N frames instead of every frame, "
        "to trade freshness for speed (default: "
        f"{StitchConfig.seam_interval} for a photo, {VideoStitchConfig.seam_interval} "
        "for a video, where the seam search is ~78%% of the per-frame cost). "
        "Combined with --seam-smoothing, an increase never causes a visible pop: the "
        "blend mask keeps easing towards the last computed seam between updates",
    )
    parser.add_argument(
        "--seam-smoothing", type=float, default=StitchConfig.seam_smoothing,
        help="for video: exponential-smoothing factor easing the blend mask towards "
        "the latest seam each frame, in (0, 1] (default: %(default)s). 1.0 snaps "
        "immediately (fine for a single image); smaller values reduce frame-to-frame "
        "seam jitter, e.g. when a subject crosses the seam band",
    )
    parser.add_argument(
        "--color-match-smoothing", type=float, default=StitchConfig.color_match_smoothing,
        help="for video: exponential-smoothing factor easing the fitted color-correction "
        "towards its latest fit each frame, in (0, 1] (default: %(default)s). Same idea "
        "as --seam-smoothing, applied to the back lens' color match instead of the seam",
    )
    parser.add_argument(
        "--auto-level", choices=["off", "vertical", "horizon", "auto"],
        default=StitchConfig.auto_level,
        help="level the panorama by detecting straight lines in the front lens' "
        "central FOV (default: %(default)s). 'vertical' uses a Manhattan-world "
        "vanishing-point solve (needs several real vertical edges, e.g. door/window "
        "frames — human-made environments); 'horizon' uses a detected horizon line "
        "(open/outdoor scenes); 'auto' tries vertical, falls back to horizon. "
        "Ignored if --level-roll/--level-pitch is set. Falls back to no leveling, "
        "with a logged reason, if no reliable cue is found — never breaks the stitch",
    )
    parser.add_argument(
        "--level-max-tilt", type=float, default=StitchConfig.level_max_tilt_deg,
        help="reject an auto-level estimate beyond this many degrees of tilt as "
        "implausible, e.g. from locking onto the wrong line family (default: %(default)s)",
    )
    parser.add_argument(
        "--level-sample-frames", type=int, default=5,
        help="video with --auto-level: estimate from this many frames spread evenly "
        "through the clip and average the result (default: %(default)s), instead of "
        "trusting a single frame — the rotation is fixed once for the whole video, so "
        "one unlucky frame would otherwise have nothing to average away. Ignored for "
        "a single photo (nothing to average over) and with --level-roll/--level-pitch",
    )
    parser.add_argument(
        "--level-dry-run", action="store_true",
        help="run auto-level detection, log the estimate, and exit without stitching "
        "anything or applying it — check what --auto-level would do before committing",
    )
    parser.add_argument(
        "--level-roll", type=float, default=None,
        help="manually level the panorama by this roll angle in degrees (rotation "
        "about the front camera's optical axis) — setting either this or "
        "--level-pitch enables leveling and takes precedence over --auto-level",
    )
    parser.add_argument(
        "--level-pitch", type=float, default=None,
        help="manually level the panorama by this pitch angle in degrees (positive "
        "= the optical axis points above the true horizon); see --level-roll",
    )
    parser.add_argument(
        "--level-fov", type=float, default=StitchConfig.level_fov_deg,
        help="field of view in degrees of the front lens' central patch used for "
        "line-based orientation estimation (default: %(default)s)",
    )
    parser.add_argument(
        "--level-patch-size", type=int, default=StitchConfig.level_patch_size,
        help="pixel width/height of that patch (default: %(default)s)",
    )
    parser.add_argument(
        "--level-projection", choices=["rectilinear", "cylindrical"],
        default=StitchConfig.level_projection,
        help="projection used for the patch (default: %(default)s — every straight "
        "3D line stays straight in 2D, which is what the line-based estimator needs; "
        "'cylindrical' is not usable with --auto-level, diagnostic preview only)",
    )
    parser.add_argument(
        "--level-preview", type=Path, default=None,
        help="write the extracted central-FOV patch (see --level-fov/--level-patch-size/"
        "--level-projection) to this path and exit, without stitching anything — a "
        "diagnostic to check the patch geometry",
    )
    parser.add_argument(
        "--width", type=int, default=None,
        help="output panorama width in pixels; must be (and will be rounded down to) "
        f"a multiple of 2**blend-levels (default: {StitchConfig.output_width} for a photo, "
        f"{VideoStitchConfig.output_width} for a video — matched to the camera's own "
        "video resolution, see VideoStitchConfig. Lowering it below that default costs "
        "real detail; it does leave memory for more --workers, since each holds a full "
        "copy of the direction/vignette/overlap caches sized to this resolution)",
    )
    parser.add_argument(
        "--height", type=int, default=None,
        help="output panorama height in pixels (default: width // 2)",
    )
    parser.add_argument(
        "--blend-levels", type=int, default=StitchConfig.blend_levels,
        help="number of Laplacian pyramid levels used for multiband blending",
    )
    parser.add_argument(
        "--baseline-scale", type=float, default=StitchConfig.baseline_scale,
        help="front-back lens separation in METRES (default: %(default)s). The "
        "extrinsics file's translation is unit-norm, so this sets its length "
        "outright -- it is not a unit conversion and is unrelated to the "
        "calibration board's square size",
    )
    parser.add_argument(
        "--depth", type=float, default=StitchConfig.depth,
        help="assumed scene depth in metres, used to reproject the back lens "
        "through the extrinsics; only its ratio to --baseline-scale matters. "
        "Note that the value which minimises seam artifacts need not be a "
        "real scene distance: it also absorbs a constant angular offset "
        "between the two lenses' fitted models",
    )
    parser.add_argument(
        "--codec", default="libx264",
        help="video: ffmpeg encoder for the output (default: %(default)s; e.g. "
        "libx265 for smaller files, which not every player decodes). Frames are "
        "piped straight into ffmpeg, so each is compressed once",
    )
    parser.add_argument(
        "--crf", type=int, default=18,
        help="video: constant-rate-factor quality, lower is better/larger "
        "(default: %(default)s, visually near-lossless)",
    )
    parser.add_argument(
        "--max-frames", type=int, default=None,
        help="video: stop after this many frames (for a quick preview)",
    )
    parser.add_argument(
        "--progress-interval", type=int, default=30,
        help="video: log progress every N frames (0 to disable)",
    )
    parser.add_argument(
        "--no-audio", action="store_true",
        help="video: don't copy the original audio track into the output "
        "(by default it's remuxed in via ffmpeg, unchanged, after stitching)",
    )
    parser.add_argument(
        "--workers", type=int, default=VideoStitchConfig.DEFAULT_WORKERS,
        help="video: stitch this many frame-range chunks in parallel processes "
        "(default: %(default)s = one per core, capped by available memory; 1 forces a "
        "single process). One frame's cv2 work already uses all cores internally but "
        "leaves most of them idle most of the time — a single-process run measures "
        "~217%% CPU on 16 cores — so this is the main lever for wall-clock speedup; "
        "needs ffmpeg to concatenate the per-chunk output afterwards (falls back to a "
        "single process without it)",
    )
    parser.add_argument(
        "--warmup-frames", type=int, default=VideoStitchConfig.DEFAULT_WARMUP_FRAMES,
        help="video with --workers > 1: replay this many frames before each chunk's "
        "start (without writing them) so its temporal seam/color-match smoothing has "
        "already converged by the time its real output begins, avoiding a visible pop "
        "at the boundary between chunks (default: %(default)s)",
    )
    parser.add_argument(
        "-v", "--verbose", action="store_true", help="enable debug logging"
    )
    return parser.parse_args(argv)


def _load_front_camera_and_frame(args: argparse.Namespace, is_video: bool):
    """The front lens' calibration (scaled to the actual frame resolution)
    and its half of one representative frame — the photo itself, or a
    video's first frame. Shared by --level-preview and --level-dry-run,
    neither of which stitch anything."""
    front_cam = FisheyeCamera.load(args.calib_front)
    if is_video:
        cap = cv2.VideoCapture(str(args.input))
        if not cap.isOpened():
            raise FileNotFoundError(f"Could not open video: {args.input}")
        ok, frame = cap.read()
        cap.release()
        if not ok:
            raise RuntimeError(f"Could not read a frame from {args.input}")
        image = frame.astype(np.float32) / 255.0
    else:
        image = load_image(args.input)
    front_im, _ = split_dual_fisheye(image)
    return front_cam.scaled_to(front_im.shape[:2]), front_im


def _write_level_preview(args: argparse.Namespace, is_video: bool) -> None:
    """Extract and save the central-FOV patch used for line-based
    orientation estimation, without stitching anything — lets the patch
    geometry (--level-fov/--level-patch-size/--level-projection) be
    eyeballed before detection is wired up on top of it."""
    front_cam, front_im = _load_front_camera_and_frame(args, is_video)
    dirs, _, _ = patch_directions(args.level_fov, args.level_patch_size, args.level_projection)
    patch = render_center_patch(front_im, front_cam, dirs)
    args.level_preview.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(args.level_preview), patch)
    logger.info("Wrote %s (%dx%d %s patch, %.0f° FOV)",
                args.level_preview, patch.shape[1], patch.shape[0],
                args.level_projection, args.level_fov)


def _run_level_dry_run(args: argparse.Namespace, is_video: bool, config: StitchConfig) -> None:
    """Run auto-level detection and log the estimate (estimate_level_rotation(_from_video)
    already logs the details), without stitching anything or applying it —
    check what --auto-level would do first. For a video this samples/averages
    the same way the real stitch would (see --level-sample-frames), not just
    the first frame, so the dry-run number matches what would actually be applied."""
    if config.auto_level == "off":
        config = dataclasses.replace(config, auto_level="auto")
    if is_video:
        front_cam = FisheyeCamera.load(args.calib_front)
        estimate_level_rotation_from_video(args.input, front_cam, config, args.level_sample_frames)
    else:
        front_cam, front_im = _load_front_camera_and_frame(args, is_video)
        estimate_level_rotation(front_cam, front_im, config)
    logger.info("Dry run: nothing stitched, no rotation applied")


def main(argv=None) -> int:
    args = parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s: %(message)s",
    )

    is_video = args.input.suffix.lower() in VIDEO_EXTENSIONS

    if args.level_preview is not None:
        _write_level_preview(args, is_video)
        return 0

    default_ext = args.input.suffix if is_video else ".jpg"
    output = args.output or args.input.with_name(f"{args.input.stem}_equirect{default_ext}")

    # Every default that differs between a photo and a video lives in
    # VideoStitchConfig (which documents why each one differs), not here:
    # the options below take None when unset precisely so the class, and
    # not this function, gets to decide them.
    config_cls = VideoStitchConfig if is_video else StitchConfig
    overrides = {
        name: value
        for name, value in (
            ("output_width", args.width),
            ("anti_alias", args.anti_alias),
            ("post_process", args.post_process),
            ("seam_interval", args.seam_interval),
            ("seam_algorithm", args.seam_algorithm),
        )
        if value is not None
    }

    config = config_cls(
        output_height=args.height,
        mask_mode=args.mask_mode,
        post_process_denoise=args.post_process_denoise,
        post_process_contrast=args.post_process_contrast,
        post_process_sharpen=args.post_process_sharpen,
        blend_levels=args.blend_levels,
        baseline_scale=args.baseline_scale,
        depth=args.depth,
        seam_smoothing=args.seam_smoothing,
        color_match_smoothing=args.color_match_smoothing,
        level_roll_deg=args.level_roll,
        level_pitch_deg=args.level_pitch,
        level_fov_deg=args.level_fov,
        level_patch_size=args.level_patch_size,
        level_projection=args.level_projection,
        auto_level=args.auto_level,
        level_max_tilt_deg=args.level_max_tilt,
        **overrides,
    )

    if args.level_dry_run:
        _run_level_dry_run(args, is_video, config)
        return 0

    if is_video:
        n = stitch_video_parallel(
            args.input, output,
            args.calib_front, args.calib_back, args.extrinsics, config,
            workers=args.workers, codec=args.codec, crf=args.crf, max_frames=args.max_frames,
            warmup_frames=args.warmup_frames, copy_audio=not args.no_audio,
            progress_interval=args.progress_interval,
            level_sample_frames=args.level_sample_frames,
        )
        tag_equirectangular_video(output)
        logger.info("Wrote %d frames to %s", n, output)
    else:
        stitcher = Gear360Stitcher.from_calibration(
            args.calib_front, args.calib_back, args.extrinsics, config
        )
        image = load_image(args.input)
        panorama = stitcher.stitch(image)
        save_image(output, panorama)
        tag_equirectangular(output, panorama.shape[1], panorama.shape[0])
        logger.info("Wrote %s", output)
    return 0


if __name__ == "__main__":
    sys.exit(main())
