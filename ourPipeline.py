import sys
import os
import subprocess
import shutil
from pathlib import Path
import numpy as np
import json
import cv2
import argparse


# =============================================================================
# ROOT PATHS
# =============================================================================

ROOT = Path(r"D:\INGP")
TESTDATA = ROOT / "testdata"
INSTANT_NGP = ROOT / "instant-ngp.exe"


# =============================================================================
# DEFAULT SAM 3 PATH
# =============================================================================

DEFAULT_SAM3_PATH = Path(
    r"D:\models\Segmentation\sam3.pt"
)


# =============================================================================
# CAMERA CONFIGURATIONS
#
# Camera filenames are matched case-insensitively.
#
# Internally, all camera names are normalized to lowercase.
# =============================================================================

CAMERA_CONFIGURATIONS = {
    4: {
        "cameras": [
            "top_left",
            "top_right",
            "bottom_right",
            "bottom_left",
        ],
        "pairs": [
            ("top_left", "top_right"),
            ("top_right", "bottom_right"),
            ("bottom_right", "bottom_left"),
            ("bottom_left", "top_left"),
        ],
    },

    6: {
        "cameras": [
            "top_left",
            "top_right",
            "center_right",
            "bottom_right",
            "bottom_left",
            "center_left",
        ],
        "pairs": [
            ("top_left", "top_right"),
            ("top_right", "center_right"),
            ("center_right", "bottom_right"),
            ("bottom_right", "bottom_left"),
            ("bottom_left", "center_left"),
            ("center_left", "top_left"),
        ],
    },

    8: {
        "cameras": [
            "top_left",
            "top_center",
            "top_right",
            "center_right",
            "bottom_right",
            "bottom_center",
            "bottom_left",
            "center_left",
        ],
        "pairs": [
            ("top_left", "top_center"),
            ("top_center", "top_right"),
            ("top_right", "center_right"),
            ("center_right", "bottom_right"),
            ("bottom_right", "bottom_center"),
            ("bottom_center", "bottom_left"),
            ("bottom_left", "center_left"),
            ("center_left", "top_left"),
        ],
    },
}


SUPPORTED_CAMERA_NAMES = {
    camera_name
    for configuration in CAMERA_CONFIGURATIONS.values()
    for camera_name in configuration["cameras"]
}


# =============================================================================
# SAM 3
# =============================================================================

SAM3_CONF = 0.25
SAM3_QUANTIZE = 16


# =============================================================================
# ALIKED
# =============================================================================

ALIKED_MAX_NUM_FEATURES = 2048
ALIKED_MIN_SCORE = 0.2


# =============================================================================
# SIFT
# =============================================================================

SIFT_MAX_NUM_FEATURES = 8192


# =============================================================================
# MATCHING
# =============================================================================

USE_GPU = "1"
GUIDED_MATCHING = "0"


# =============================================================================
# COLMAP MAPPER
# =============================================================================

MAPPER_MULTIPLE_MODELS = "1"
MAPPER_STRUCTURE_LESS_FALLBACK = "1"
MAPPER_MIN_NUM_MATCHES = "15"
MAPPER_INIT_MIN_NUM_INLIERS = "100"
MAPPER_ABS_POSE_MIN_NUM_INLIERS = "30"
MAPPER_ABS_POSE_MIN_INLIER_RATIO = "0.25"
MAPPER_INIT_MAX_ERROR = "4"
MAPPER_FILTER_MAX_REPROJ_ERROR = "4"
MAPPER_INIT_MIN_TRI_ANGLE = "16"


# =============================================================================
# ARGUMENT PARSER
# =============================================================================

def parse_args():

    parser = argparse.ArgumentParser(
        description=(
            "4/6/8-camera SAM 3 -> RGBA -> COLMAP -> "
            "Instant-NGP reconstruction pipeline."
        )
    )

    parser.add_argument(
        "--project",
        required=True,
        help="Project folder name inside D:\\INGP\\testdata.",
    )

    parser.add_argument(
        "--frames",
        type=int,
        default=None,
        help=(
            "Number of frames per video. "
            "If omitted, approximately 2 FPS is used."
        ),
    )

    parser.add_argument(
        "--prompt",
        required=True,
        help="SAM 3 segmentation prompt.",
    )

    parser.add_argument(
        "--sam3",
        type=Path,
        default=DEFAULT_SAM3_PATH,
        help="Path to SAM 3 model.",
    )

    parser.add_argument(
        "--colmap",
        type=Path,
        default=None,
        help=(
            "Optional path to COLMAP. "
            "If omitted, COLMAP is resolved from PATH."
        ),
    )

    parser.add_argument(
        "--feature",
        choices=[
            "aliked",
            "sift",
        ],
        default="aliked",
        help="Feature extractor.",
    )

    parser.add_argument(
        "--matcher",
        choices=[
            "lightglue",
            "bruteforce",
        ],
        default="lightglue",
        help="Feature matcher.",
    )

    return parser.parse_args()


# =============================================================================
# FIND COLMAP
# =============================================================================

def find_colmap(colmap_argument=None):

    # -------------------------------------------------------------------------
    # Explicit --colmap argument
    # -------------------------------------------------------------------------

    if colmap_argument is not None:

        colmap = Path(colmap_argument)

        if not colmap.exists():
            raise RuntimeError(
                f"COLMAP executable does not exist:\n{colmap}"
            )

        return colmap

    # -------------------------------------------------------------------------
    # Resolve COLMAP from Windows PATH
    # -------------------------------------------------------------------------

    colmap = shutil.which("colmap")

    if colmap is None:
        colmap = shutil.which("COLMAP.bat")

    if colmap is None:
        colmap = shutil.which("colmap.bat")

    if colmap is None:
        colmap = shutil.which("COLMAP.exe")

    if colmap is None:
        raise RuntimeError(
            "COLMAP was not found on PATH.\n"
            "Run 'where colmap' in the same terminal to verify PATH."
        )

    return Path(colmap)


# =============================================================================
# EXTERNAL COMMAND EXECUTION
# =============================================================================

def run(*args):

    command = [
        str(arg)
        for arg in args
    ]

    # -------------------------------------------------------------------------
    # Windows .BAT/.CMD handling
    # -------------------------------------------------------------------------

    if (
        command
        and command[0].lower().endswith(
            (".bat", ".cmd")
        )
    ):
        command = [
            os.environ.get(
                "ComSpec",
                "cmd.exe",
            ),
            "/c",
            *command,
        ]

    print()
    print("=" * 80)
    print("RUNNING:")
    print(
        " ".join(
            f'"{x}"' if " " in x else x
            for x in command
        )
    )
    print("=" * 80)

    subprocess.run(
        command,
        check=True,
    )


# =============================================================================
# GET MATCHER TYPE
# =============================================================================

def get_matcher_type(
    feature,
    matcher,
):

    if feature == "aliked":

        if matcher == "lightglue":
            return "ALIKED_LIGHTGLUE"

        if matcher == "bruteforce":
            return "ALIKED_BRUTEFORCE"

    elif feature == "sift":

        if matcher == "lightglue":
            return "SIFT_LIGHTGLUE"

        if matcher == "bruteforce":
            return "SIFT_BRUTEFORCE"

    raise RuntimeError(
        f"Unsupported feature/matcher combination: "
        f"{feature} + {matcher}"
    )


# =============================================================================
# FIND SOURCE VIDEOS
# =============================================================================

def find_source_videos(
    project,
):
    """
    Find camera videos case-insensitively.

    Examples accepted:
        top_left.mp4
        Top_Left.mp4
        TOP_LEFT.MP4

    The returned dictionary always uses canonical lowercase camera names.
    """

    videos = {}

    for entry in project.iterdir():

        if not entry.is_file():
            continue

        if entry.suffix.lower() != ".mp4":
            continue

        stem = entry.stem.lower()

        if stem not in SUPPORTED_CAMERA_NAMES:
            continue

        if stem in videos:
            raise RuntimeError(
                "Duplicate camera video detected for "
                f"'{stem}':\n"
                f"{videos[stem]}\n"
                f"{entry}"
            )

        videos[stem] = entry

    return videos


# =============================================================================
# DETECT CAMERA CONFIGURATION
# =============================================================================

def detect_camera_setup(
    project,
):
    """
    Detect exactly one valid 4-, 6-, or 8-camera arrangement.

    Video filename capitalization does not matter.
    Extra unrelated MP4 files are ignored.
    """

    videos = find_source_videos(
        project
    )

    present = set(
        videos.keys()
    )

    camera_count = len(
        present
    )

    if camera_count not in CAMERA_CONFIGURATIONS:
        raise RuntimeError(
            f"Detected {camera_count} recognized camera videos. "
            "Exactly 4, 6, or 8 camera videos are required."
        )

    configuration = CAMERA_CONFIGURATIONS[
        camera_count
    ]

    expected = set(
        configuration["cameras"]
    )

    if present != expected:

        missing = sorted(
            expected - present
        )

        unexpected = sorted(
            present - expected
        )

        message = (
            "The camera videos do not form a valid "
            "4-, 6-, or 8-camera configuration."
        )

        if missing:
            message += (
                "\n\nMissing expected cameras:\n"
                +
                "\n".join(
                    f"  {camera}.mp4"
                    for camera in missing
                )
            )

        if unexpected:
            message += (
                "\n\nUnexpected recognized cameras:\n"
                +
                "\n".join(
                    f"  {camera}.mp4"
                    for camera in unexpected
                )
            )

        raise RuntimeError(
            message
        )

    return (
        camera_count,
        configuration["cameras"],
        configuration["pairs"],
        videos,
    )


# =============================================================================
# VIDEO INFORMATION
# =============================================================================

def video_info(
    video,
):

    cap = cv2.VideoCapture(
        str(video)
    )

    if not cap.isOpened():
        raise RuntimeError(
            f"Could not open video:\n{video}"
        )

    fps = cap.get(
        cv2.CAP_PROP_FPS
    )

    total_frames = int(
        cap.get(
            cv2.CAP_PROP_FRAME_COUNT
        )
    )

    cap.release()

    if fps <= 0 or total_frames <= 0:
        raise RuntimeError(
            f"Could not determine FPS/frame count:\n{video}"
        )

    duration = (
        total_frames /
        fps
    )

    # Approximately 2 FPS.

    default_frames = max(
        1,
        round(duration * 2.0)
    )

    default_frames = min(
        default_frames,
        total_frames
    )

    return (
        fps,
        total_frames,
        duration,
        default_frames,
    )


# =============================================================================
# FRAME SELECTION
# =============================================================================

def get_frame_indices(
    total_frames,
    count,
):

    if count < 1:
        raise RuntimeError(
            "Frame count must be at least 1."
        )

    if count > total_frames:
        raise RuntimeError(
            f"Requested {count} frames, "
            f"but video contains only {total_frames} frames."
        )

    if count == 1:
        return np.array(
            [0],
            dtype=np.int64,
        )

    return np.linspace(
        0,
        total_frames - 1,
        count,
    ).round().astype(
        np.int64
    )


# =============================================================================
# COMBINE SAM 3 MASKS
# =============================================================================

def create_combined_mask(
    results,
    frame,
):

    if not (
        results
        and results[0].masks is not None
        and len(results[0].masks.data)
    ):
        return (
            None,
            0,
        )

    masks = (
        results[0]
        .masks
        .data
        .cpu()
        .numpy()
    )

    height, width = frame.shape[:2]

    combined_mask = np.zeros(
        (height, width),
        dtype=np.uint8,
    )

    for mask in masks:

        if mask.shape != (
            height,
            width,
        ):
            mask = cv2.resize(
                mask,
                (width, height),
                interpolation=cv2.INTER_NEAREST,
            )

        combined_mask = np.maximum(
            combined_mask,
            (
                mask > 0
            ).astype(
                np.uint8
            ) * 255,
        )

    return (
        combined_mask,
        len(masks),
    )


# =============================================================================
# PROCESS ONE VIDEO
# =============================================================================

def create_rgba_from_video(
    video,
    output_dir,
    count,
    predictor,
    prompt,
):

    if output_dir.exists():
        shutil.rmtree(
            output_dir
        )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    (
        fps,
        total_frames,
        duration,
        default_frames,
    ) = video_info(
        video
    )

    selected_count = (
        default_frames
        if count is None
        else count
    )

    print()
    print("=" * 80)
    print(
        f"VIDEO: {video.name}"
    )
    print("=" * 80)

    print(
        f"FPS:          {fps:.3f}"
    )

    print(
        f"Total frames: {total_frames}"
    )

    print(
        f"Duration:     {duration:.2f} seconds"
    )

    if count is None:

        print(
            f"Frames used:  {selected_count} "
            f"(approximately 2 FPS)"
        )

    else:

        print(
            f"Frames used:  {selected_count}"
        )

    indices = get_frame_indices(
        total_frames,
        selected_count,
    )

    wanted = {
        int(frame_index): index + 1
        for index, frame_index
        in enumerate(indices)
    }

    cap = cv2.VideoCapture(
        str(video)
    )

    if not cap.isOpened():
        raise RuntimeError(
            f"Could not open video:\n{video}"
        )

    frame_idx = 0
    written = 0

    while True:

        ret, frame = cap.read()

        if not ret:
            break

        if frame_idx in wanted:

            out_idx = wanted[
                frame_idx
            ]

            predictor.set_image(
                frame
            )

            results = predictor(
                text=[prompt]
            )

            (
                alpha,
                object_count,
            ) = create_combined_mask(
                results,
                frame,
            )

            if alpha is None:

                print(
                    f"Frame {out_idx:04d}: "
                    f"WARNING - no mask found"
                )

            else:

                # -------------------------------------------------------------
                # BGR + alpha -> four-channel PNG
                #
                # Object:
                #     alpha = 255
                #
                # Background:
                #     alpha = 0
                #
                # RGB/BGR values are preserved.
                #
                # No separate binary mask is stored.
                # -------------------------------------------------------------

                rgba = np.dstack(
                    (
                        frame,
                        alpha,
                    )
                )

                output = (
                    output_dir /
                    f"{out_idx:04d}.png"
                )

                if not cv2.imwrite(
                    str(output),
                    rgba,
                ):
                    raise RuntimeError(
                        f"Could not write:\n{output}"
                    )

                written += 1

                print(
                    f"Frame {out_idx:04d}: "
                    f"{object_count} matching object(s)"
                )

        frame_idx += 1

        if frame_idx > indices[-1]:
            break

    cap.release()

    print()

    print(
        f"Created {written} RGBA images."
    )

    if written == 0:
        raise RuntimeError(
            f"SAM 3 produced no RGBA images "
            f"for {video.name}"
        )


# =============================================================================
# FRESH PROJECT CLEANUP
# =============================================================================

def clean_project(
    project,
    source_videos,
):
    """
    Delete everything in the project directory except the actual source
    camera MP4 files detected for the current 4/6/8-camera setup.

    Preserved:
        source camera MP4 files

    Deleted:
        images/**
        database.db
        image_pairs.txt
        sparse/**
        colmap_text/**
        transforms.json
        every other file/directory
    """

    print()
    print("=" * 80)
    print("FRESH PROJECT CLEANUP")
    print("=" * 80)

    preserved = {
        path.resolve()
        for path in source_videos.values()
    }

    for child in project.iterdir():

        if child.resolve() in preserved:

            print(
                f"PRESERVE: {child.name}"
            )

            continue

        if child.is_dir():

            print(
                f"DELETE:   {child}"
            )

            shutil.rmtree(
                child
            )

        else:

            print(
                f"DELETE:   {child}"
            )

            child.unlink()

    print()

    print(
        "Fresh project cleanup complete."
    )


# =============================================================================
# LOAD GENERATED CAMERA IMAGES
# =============================================================================

def get_camera_images(
    project,
    camera,
):

    camera_dir = (
        project /
        "images" /
        camera
    )

    if not camera_dir.exists():
        raise RuntimeError(
            f"Camera image directory does not exist:\n"
            f"{camera_dir}"
        )

    image_paths = sorted(
        camera_dir.glob("*.png")
    )

    if not image_paths:
        raise RuntimeError(
            f"No PNG images found in:\n"
            f"{camera_dir}"
        )

    return image_paths


def load_camera_images(
    project,
    active_cameras,
):

    camera_images = {}

    print()
    print("=" * 80)
    print("RGBA DATASET")
    print("=" * 80)

    total = 0

    for camera in active_cameras:

        image_paths = get_camera_images(
            project,
            camera,
        )

        camera_images[
            camera
        ] = image_paths

        total += len(
            image_paths
        )

        print(
            f"{camera:15s}: "
            f"{len(image_paths):4d} images"
        )

    print("-" * 80)

    print(
        f"Total images: {total}"
    )

    return camera_images


# =============================================================================
# VALIDATE RGBA
# =============================================================================

def validate_rgba_images(
    camera_images,
):

    print()
    print("=" * 80)
    print("VALIDATING RGBA IMAGES")
    print("=" * 80)

    total = 0

    for camera in camera_images:

        for image_path in camera_images[
            camera
        ]:

            image = cv2.imread(
                str(image_path),
                cv2.IMREAD_UNCHANGED,
            )

            if image is None:
                raise RuntimeError(
                    f"Could not read:\n{image_path}"
                )

            if (
                len(image.shape) != 3
                or image.shape[2] != 4
            ):
                raise RuntimeError(
                    f"Expected 4-channel PNG:\n"
                    f"{image_path}\n"
                    f"Actual shape: {image.shape}"
                )

            alpha = image[:, :, 3]

            if not np.any(
                alpha > 0
            ):
                raise RuntimeError(
                    f"Alpha channel is completely empty:\n"
                    f"{image_path}"
                )

            total += 1

    print(
        f"Validated {total} RGBA images."
    )


# =============================================================================
# RELATIVE IMAGE NAME
# =============================================================================

def relative_image_name(
    image_path,
    project,
):

    return str(
        image_path.relative_to(
            project /
            "images"
        )
    ).replace(
        "\\",
        "/"
    )


# =============================================================================
# ADD UNIQUE PAIR
# =============================================================================

def add_pair(
    pair_set,
    image_a,
    image_b,
    project,
):

    name_a = relative_image_name(
        image_a,
        project,
    )

    name_b = relative_image_name(
        image_b,
        project,
    )

    if name_a == name_b:
        return

    if name_a < name_b:

        pair_set.add(
            (
                name_a,
                name_b,
            )
        )

    else:

        pair_set.add(
            (
                name_b,
                name_a,
            )
        )


# =============================================================================
# CREATE EXACT MATCHING GRAPH
# =============================================================================

def create_image_pairs(
    project,
    camera_images,
    camera_pairs,
):

    pair_set = set()

    # =========================================================================
    # WITHIN-CAMERA ALL-VS-ALL
    # =========================================================================

    print()
    print("=" * 80)
    print("WITHIN-CAMERA ALL-VS-ALL")
    print("=" * 80)

    within_total = 0

    for camera in camera_images:

        images = camera_images[
            camera
        ]

        before = len(
            pair_set
        )

        for i in range(
            len(images)
        ):

            for j in range(
                i + 1,
                len(images)
            ):

                add_pair(
                    pair_set,
                    images[i],
                    images[j],
                    project,
                )

        added = (
            len(pair_set) -
            before
        )

        expected = (
            len(images) *
            (len(images) - 1)
            // 2
        )

        if added != expected:
            raise RuntimeError(
                f"Within-camera pair mismatch for "
                f"{camera}: expected {expected}, got {added}"
            )

        within_total += added

        print(
            f"{camera:15s}: "
            f"{len(images):4d} images -> "
            f"{added:6d} pairs"
        )

    # =========================================================================
    # BETWEEN-CAMERA ALL-VS-ALL
    # =========================================================================

    print()
    print("=" * 80)
    print("BETWEEN-CAMERA ALL-VS-ALL")
    print("=" * 80)

    cross_total = 0

    for camera_a, camera_b in camera_pairs:

        images_a = camera_images[
            camera_a
        ]

        images_b = camera_images[
            camera_b
        ]

        before = len(
            pair_set
        )

        for image_a in images_a:

            for image_b in images_b:

                add_pair(
                    pair_set,
                    image_a,
                    image_b,
                    project,
                )

        added = (
            len(pair_set) -
            before
        )

        expected = (
            len(images_a) *
            len(images_b)
        )

        if added != expected:
            raise RuntimeError(
                f"Cross-camera pair mismatch for "
                f"{camera_a} <-> {camera_b}: "
                f"expected {expected}, got {added}"
            )

        cross_total += added

        print(
            f"{camera_a:15s} <-> "
            f"{camera_b:15s}: "
            f"{len(images_a):4d} x "
            f"{len(images_b):4d} = "
            f"{added:6d}"
        )

    # =========================================================================
    # VERIFY TOTAL
    # =========================================================================

    expected_within_total = sum(
        len(camera_images[camera]) *
        (
            len(camera_images[camera]) - 1
        )
        // 2
        for camera in camera_images
    )

    expected_cross_total = sum(
        len(camera_images[camera_a]) *
        len(camera_images[camera_b])
        for camera_a, camera_b in camera_pairs
    )

    expected_total = (
        expected_within_total +
        expected_cross_total
    )

    if within_total != expected_within_total:
        raise RuntimeError(
            f"Within-camera total mismatch: "
            f"expected {expected_within_total}, "
            f"got {within_total}"
        )

    if cross_total != expected_cross_total:
        raise RuntimeError(
            f"Cross-camera total mismatch: "
            f"expected {expected_cross_total}, "
            f"got {cross_total}"
        )

    if len(pair_set) != expected_total:
        raise RuntimeError(
            f"Total pair mismatch: "
            f"expected {expected_total}, "
            f"got {len(pair_set)}"
        )

    # =========================================================================
    # WRITE PAIR FILE
    # =========================================================================

    pairs = sorted(
        pair_set
    )

    pair_file = (
        project /
        "image_pairs.txt"
    )

    with open(
        pair_file,
        "w",
        encoding="utf-8",
    ) as f:

        for image_a, image_b in pairs:

            f.write(
                f"{image_a} {image_b}\n"
            )

    print()
    print("-" * 80)

    print(
        f"Within-camera pairs: {within_total}"
    )

    print(
        f"Cross-camera pairs:  {cross_total}"
    )

    print(
        f"Total unique pairs:  {len(pairs)}"
    )

    print(
        f"Pair file:            {pair_file}"
    )

    return pair_file


# =============================================================================
# COLMAP FEATURE EXTRACTION
# =============================================================================

def run_feature_extraction(
    colmap,
    image_dir,
    database,
    feature,
):

    print()
    print("=" * 80)
    print("COLMAP FEATURE EXTRACTION")
    print("=" * 80)

    print(
        "Extractor:",
        "ALIKED-N32"
        if feature == "aliked"
        else "SIFT",
    )

    command = [
        colmap,
        "feature_extractor",
        "--database_path",
        database,
        "--image_path",
        image_dir,
        "--ImageReader.single_camera_per_folder",
        "1",
        "--ImageReader.camera_model",
        "OPENCV",
        "--FeatureExtraction.type",
        (
            "ALIKED_N32"
            if feature == "aliked"
            else "SIFT"
        ),
        "--FeatureExtraction.use_gpu",
        USE_GPU,
    ]

    if feature == "aliked":

        command.extend([
            "--AlikedExtraction.max_num_features",
            str(
                ALIKED_MAX_NUM_FEATURES
            ),
            "--AlikedExtraction.min_score",
            str(
                ALIKED_MIN_SCORE
            ),
        ])

    else:

        command.extend([
            "--SiftExtraction.max_num_features",
            str(
                SIFT_MAX_NUM_FEATURES
            ),
        ])

    run(
        *command
    )


# =============================================================================
# COLMAP FEATURE MATCHING
# =============================================================================

def run_feature_matching(
    colmap,
    database,
    pair_file,
    feature,
    matcher,
):

    matcher_type = get_matcher_type(
        feature,
        matcher,
    )

    print()
    print("=" * 80)
    print("COLMAP FEATURE MATCHING")
    print("=" * 80)

    print(
        f"Matcher: {matcher_type}"
    )

    run(
        colmap,
        "matches_importer",
        "--database_path",
        database,
        "--match_list_path",
        pair_file,
        "--match_type",
        "pairs",
        "--FeatureMatching.type",
        matcher_type,
        "--FeatureMatching.use_gpu",
        USE_GPU,
        "--FeatureMatching.guided_matching",
        GUIDED_MATCHING,
    )


# =============================================================================
# COLMAP MAPPER
# =============================================================================

def run_mapper(
    colmap,
    image_dir,
    database,
    sparse_dir,
):

    print()
    print("=" * 80)
    print("COLMAP RECONSTRUCTION")
    print("=" * 80)

    sparse_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    run(
        colmap,
        "mapper",
        "--database_path",
        database,
        "--image_path",
        image_dir,
        "--output_path",
        sparse_dir,
        "--Mapper.multiple_models",
        MAPPER_MULTIPLE_MODELS,
        "--Mapper.structure_less_registration_fallback",
        MAPPER_STRUCTURE_LESS_FALLBACK,
        "--Mapper.min_num_matches",
        MAPPER_MIN_NUM_MATCHES,
        "--Mapper.init_min_num_inliers",
        MAPPER_INIT_MIN_NUM_INLIERS,
        "--Mapper.abs_pose_min_num_inliers",
        MAPPER_ABS_POSE_MIN_NUM_INLIERS,
        "--Mapper.abs_pose_min_inlier_ratio",
        MAPPER_ABS_POSE_MIN_INLIER_RATIO,
        "--Mapper.init_max_error",
        MAPPER_INIT_MAX_ERROR,
        "--Mapper.filter_max_reproj_error",
        MAPPER_FILTER_MAX_REPROJ_ERROR,
        "--Mapper.init_min_tri_angle",
        MAPPER_INIT_MIN_TRI_ANGLE,
    )


# =============================================================================
# COUNT REGISTERED IMAGES
# =============================================================================

def count_registered_images(
    images_txt,
):

    with open(
        images_txt,
        "r",
        encoding="utf-8",
    ) as f:

        lines = f.readlines()

    count = 0
    i = 0

    while i < len(lines):

        line = lines[i].strip()

        if (
            not line
            or line.startswith("#")
        ):

            i += 1
            continue

        elements = line.split()

        if len(elements) >= 10:

            count += 1
            i += 2

        else:

            i += 1

    return count


# =============================================================================
# FIND BEST MODEL
# =============================================================================

def find_best_model(
    colmap,
    sparse_dir,
    text_dir,
):

    if not sparse_dir.exists():
        raise RuntimeError(
            "COLMAP sparse directory was not created."
        )

    models = []

    for model_dir in sorted(
        sparse_dir.iterdir()
    ):

        if not model_dir.is_dir():
            continue

        if not (
            model_dir /
            "images.bin"
        ).exists():
            continue

        model_text = (
            text_dir /
            model_dir.name
        )

        if model_text.exists():
            shutil.rmtree(
                model_text
            )

        model_text.mkdir(
            parents=True,
            exist_ok=True,
        )

        run(
            colmap,
            "model_converter",
            "--input_path",
            model_dir,
            "--output_path",
            model_text,
            "--output_type",
            "TXT",
        )

        images_txt = (
            model_text /
            "images.txt"
        )

        if not images_txt.exists():
            continue

        registered = count_registered_images(
            images_txt
        )

        models.append(
            (
                model_dir,
                model_text,
                registered,
            )
        )

    if not models:
        raise RuntimeError(
            "COLMAP created no usable reconstruction model."
        )

    models.sort(
        key=lambda x: x[2],
        reverse=True,
    )

    print()
    print("=" * 80)
    print("COLMAP MODELS")
    print("=" * 80)

    for (
        model_dir,
        _,
        registered,
    ) in models:

        print(
            f"model {model_dir.name}: "
            f"{registered} registered images"
        )

    (
        best_model,
        best_text,
        best_count,
    ) = models[0]

    print()

    print(
        f"Best model:        {best_model.name}"
    )

    print(
        f"Registered images: {best_count}"
    )

    return (
        best_model,
        best_text,
        best_count,
    )


# =============================================================================
# READ COLMAP CAMERAS
# =============================================================================

def read_cameras(
    cameras_txt,
):

    cameras = {}

    with open(
        cameras_txt,
        "r",
        encoding="utf-8",
    ) as f:

        for line in f:

            line = line.strip()

            if (
                not line
                or line.startswith("#")
            ):
                continue

            elements = line.split()

            if len(elements) < 12:
                continue

            camera_id = int(
                elements[0]
            )

            model = elements[1]

            if model != "OPENCV":
                raise RuntimeError(
                    f"Expected OPENCV camera model, "
                    f"got {model} for camera {camera_id}"
                )

            width = int(
                float(elements[2])
            )

            height = int(
                float(elements[3])
            )

            fx = float(
                elements[4]
            )

            fy = float(
                elements[5]
            )

            cx = float(
                elements[6]
            )

            cy = float(
                elements[7]
            )

            k1 = float(
                elements[8]
            )

            k2 = float(
                elements[9]
            )

            p1 = float(
                elements[10]
            )

            p2 = float(
                elements[11]
            )

            cameras[
                camera_id
            ] = {
                "w": width,
                "h": height,
                "fl_x": fx,
                "fl_y": fy,
                "cx": cx,
                "cy": cy,
                "k1": k1,
                "k2": k2,
                "k3": 0.0,
                "k4": 0.0,
                "p1": p1,
                "p2": p2,
                "is_fisheye": False,
                "camera_angle_x":
                    2.0 *
                    np.arctan(
                        width /
                        (
                            2.0 *
                            fx
                        )
                    ),
                "camera_angle_y":
                    2.0 *
                    np.arctan(
                        height /
                        (
                            2.0 *
                            fy
                        )
                    ),
            }

    if not cameras:
        raise RuntimeError(
            "No cameras found in cameras.txt."
        )

    return cameras


# =============================================================================
# QUATERNION -> ROTATION MATRIX
# =============================================================================

def qvec2rotmat(
    q,
):

    w, x, y, z = q

    return np.array(
        [
            [
                1 - 2 * y * y - 2 * z * z,
                2 * x * y - 2 * w * z,
                2 * x * z + 2 * w * y,
            ],
            [
                2 * x * y + 2 * w * z,
                1 - 2 * x * x - 2 * z * z,
                2 * y * z - 2 * w * x,
            ],
            [
                2 * x * z - 2 * w * y,
                2 * y * z + 2 * w * x,
                1 - 2 * x * x - 2 * y * y,
            ],
        ],
        dtype=np.float64,
    )


# =============================================================================
# ROTATION MATRIX
# =============================================================================

def rotmat(
    a,
    b,
):

    a = np.asarray(
        a,
        dtype=np.float64,
    )

    b = np.asarray(
        b,
        dtype=np.float64,
    )

    a_norm = np.linalg.norm(
        a
    )

    b_norm = np.linalg.norm(
        b
    )

    if a_norm <= 1e-12:
        raise RuntimeError(
            "Cannot normalize zero vector."
        )

    if b_norm <= 1e-12:
        raise RuntimeError(
            "Cannot normalize zero vector."
        )

    a /= a_norm
    b /= b_norm

    v = np.cross(
        a,
        b,
    )

    c = np.dot(
        a,
        b,
    )

    if c < -1 + 1e-10:

        axis = np.array(
            [
                1.0,
                0.0,
                0.0,
            ],
            dtype=np.float64,
        )

        if abs(
            np.dot(
                a,
                axis,
            )
        ) > 0.9:

            axis = np.array(
                [
                    0.0,
                    1.0,
                    0.0,
                ],
                dtype=np.float64,
            )

        axis = np.cross(
            a,
            axis,
        )

        axis /= np.linalg.norm(
            axis
        )

        return (
            2.0 *
            np.outer(
                axis,
                axis,
            )
            -
            np.eye(3)
        )

    s = np.linalg.norm(
        v
    )

    k = np.array(
        [
            [
                0.0,
                -v[2],
                v[1],
            ],
            [
                v[2],
                0.0,
                -v[0],
            ],
            [
                -v[1],
                v[0],
                0.0,
            ],
        ],
        dtype=np.float64,
    )

    return (
        np.eye(3)
        +
        k
        +
        k @ k *
        (
            (1.0 - c) /
            (s * s + 1e-10)
        )
    )


# =============================================================================
# CLOSEST POINT BETWEEN TWO CAMERA RAYS
# =============================================================================

def closest_point_2_lines(
    oa,
    da,
    ob,
    db,
):

    da = np.asarray(
        da,
        dtype=np.float64,
    )

    db = np.asarray(
        db,
        dtype=np.float64,
    )

    da_norm = np.linalg.norm(
        da
    )

    db_norm = np.linalg.norm(
        db
    )

    if (
        da_norm <= 1e-12
        or db_norm <= 1e-12
    ):
        return (
            (oa + ob) * 0.5,
            0.0,
        )

    da /= da_norm
    db /= db_norm

    c = (
        oa -
        ob
    )

    aa = np.dot(
        da,
        da,
    )

    bb = np.dot(
        da,
        db,
    )

    cc = np.dot(
        db,
        db,
    )

    dd = np.dot(
        da,
        c,
    )

    ee = np.dot(
        db,
        c,
    )

    denom = (
        aa * cc -
        bb * bb
    )

    if abs(
        denom
    ) < 1e-10:
        return (
            (oa + ob) * 0.5,
            0.0,
        )

    ta = (
        bb * ee -
        cc * dd
    ) / denom

    tb = (
        aa * ee -
        bb * dd
    ) / denom

    point = (
        oa +
        ta * da +
        ob +
        tb * db
    ) * 0.5

    distance = np.linalg.norm(
        oa +
        ta * da -
        ob -
        tb * db
    )

    if distance <= 1e-12:
        return (
            point,
            0.0,
        )

    return (
        point,
        1.0 / distance,
    )


# =============================================================================
# CREATE TRANSFORMS.JSON
# =============================================================================

def create_transforms(
    project,
    best_text,
):

    cameras_txt = (
        best_text /
        "cameras.txt"
    )

    images_txt = (
        best_text /
        "images.txt"
    )

    cameras = read_cameras(
        cameras_txt
    )

    print()
    print("=" * 80)
    print("COLMAP CAMERA CALIBRATION")
    print("=" * 80)

    for camera_id in sorted(
        cameras
    ):

        camera = cameras[
            camera_id
        ]

        print(
            f"Camera {camera_id}: "
            f"{camera['w']} x {camera['h']} | "
            f"fx={camera['fl_x']:.6f} | "
            f"fy={camera['fl_y']:.6f} | "
            f"cx={camera['cx']:.6f} | "
            f"cy={camera['cy']:.6f}"
        )

    frames = []

    up = np.zeros(
        3,
        dtype=np.float64,
    )

    with open(
        images_txt,
        "r",
        encoding="utf-8",
    ) as f:

        lines = f.readlines()

    i = 0

    while i < len(lines):

        line = lines[i].strip()

        if (
            not line
            or line.startswith("#")
        ):

            i += 1
            continue

        elements = line.split()

        if len(elements) < 10:

            i += 1
            continue

        qvec = np.array(
            [
                float(x)
                for x in elements[1:5]
            ],
            dtype=np.float64,
        )

        tvec = np.array(
            [
                float(x)
                for x in elements[5:8]
            ],
            dtype=np.float64,
        )

        camera_id = int(
            elements[8]
        )

        image_name = " ".join(
            elements[9:]
        )

        if camera_id not in cameras:
            raise RuntimeError(
                f"Image {image_name} references "
                f"unknown camera {camera_id}"
            )

        # COLMAP stores:
        #
        # X_camera = R * X_world + t
        #
        # Invert to obtain camera-to-world.

        R = qvec2rotmat(
            qvec
        )

        w2c = np.eye(
            4,
            dtype=np.float64,
        )

        w2c[
            :3,
            :3
        ] = R

        w2c[
            :3,
            3
        ] = tvec

        c2w = np.linalg.inv(
            w2c
        )

        # Instant-NGP coordinate conversion.

        c2w[
            0:3,
            2
        ] *= -1

        c2w[
            0:3,
            1
        ] *= -1

        c2w = c2w[
            [1, 0, 2, 3],
            :
        ]

        c2w[
            2,
            :
        ] *= -1

        up += c2w[
            0:3,
            1
        ]

        relative_path = Path(
            image_name
        )

        source_image = (
            project /
            "images" /
            relative_path
        )

        if not source_image.exists():
            raise RuntimeError(
                f"Registered image does not exist:\n"
                f"{source_image}"
            )

        image = cv2.imread(
            str(source_image),
            cv2.IMREAD_UNCHANGED,
        )

        if image is None:
            raise RuntimeError(
                f"Could not read registered image:\n"
                f"{source_image}"
            )

        if (
            len(image.shape) == 3
            and image.shape[2] == 4
        ):

            gray = cv2.cvtColor(
                image,
                cv2.COLOR_BGRA2GRAY,
            )

        else:

            gray = cv2.cvtColor(
                image,
                cv2.COLOR_BGR2GRAY,
            )

        sharpness = cv2.Laplacian(
            gray,
            cv2.CV_64F,
        ).var()

        camera = cameras[
            camera_id
        ]

        frame = {

            "file_path":
                "./images/" +
                relative_path.as_posix(),

            "sharpness":
                float(
                    sharpness
                ),

            "w":
                camera["w"],

            "h":
                camera["h"],

            "fl_x":
                camera["fl_x"],

            "fl_y":
                camera["fl_y"],

            "cx":
                camera["cx"],

            "cy":
                camera["cy"],

            "k1":
                camera["k1"],

            "k2":
                camera["k2"],

            "k3":
                camera["k3"],

            "k4":
                camera["k4"],

            "p1":
                camera["p1"],

            "p2":
                camera["p2"],

            "is_fisheye":
                camera["is_fisheye"],

            "camera_angle_x":
                camera["camera_angle_x"],

            "camera_angle_y":
                camera["camera_angle_y"],

            "transform_matrix":
                c2w,
        }

        frames.append(
            frame
        )

        i += 2

    if not frames:
        raise RuntimeError(
            "No registered images found in images.txt."
        )

    print()

    print(
        f"Registered frames: {len(frames)}"
    )

    # =========================================================================
    # ALIGN UP TO +Z
    # =========================================================================

    up_norm = np.linalg.norm(
        up
    )

    if up_norm <= 1e-12:
        raise RuntimeError(
            "Could not determine camera up direction."
        )

    up /= up_norm

    R = rotmat(
        up,
        np.array(
            [
                0.0,
                0.0,
                1.0,
            ],
            dtype=np.float64,
        ),
    )

    R4 = np.eye(
        4,
        dtype=np.float64,
    )

    R4[
        :3,
        :3
    ] = R

    for frame in frames:

        frame[
            "transform_matrix"
        ] = (
            R4 @
            frame[
                "transform_matrix"
            ]
        )

    # =========================================================================
    # CENTER OF ATTENTION
    # =========================================================================

    print()

    print(
        "Computing center of attention..."
    )

    total_weight = 0.0

    total_point = np.zeros(
        3,
        dtype=np.float64,
    )

    for frame_a in frames:

        matrix_a = (
            frame_a[
                "transform_matrix"
            ][:3]
        )

        for frame_b in frames:

            matrix_b = (
                frame_b[
                    "transform_matrix"
                ][:3]
            )

            point, weight = (
                closest_point_2_lines(
                    matrix_a[:, 3],
                    matrix_a[:, 2],
                    matrix_b[:, 3],
                    matrix_b[:, 2],
                )
            )

            if weight > 0.00001:

                total_point += (
                    point *
                    weight
                )

                total_weight += weight

    if total_weight > 0:
        total_point /= total_weight

    print(
        f"Center of attention: {total_point}"
    )

    # =========================================================================
    # CENTER
    # =========================================================================

    for frame in frames:

        frame[
            "transform_matrix"
        ][:3, 3] -= total_point

    # =========================================================================
    # SCALE
    # =========================================================================

    distances = [

        np.linalg.norm(
            frame[
                "transform_matrix"
            ][:3, 3]
        )

        for frame in frames
    ]

    average_distance = np.mean(
        distances
    )

    if average_distance <= 1e-12:
        raise RuntimeError(
            "Average camera distance is zero."
        )

    scale = (
        4.0 /
        average_distance
    )

    print(
        f"Average camera distance: {average_distance}"
    )

    print(
        f"Scale factor: {scale}"
    )

    for frame in frames:

        frame[
            "transform_matrix"
        ][:3, 3] *= scale

        frame[
            "transform_matrix"
        ] = (
            frame[
                "transform_matrix"
            ].tolist()
        )

    return {
        "aabb_scale": 4,
        "frames": frames,
    }


# =============================================================================
# RUN INSTANT-NGP
# =============================================================================

def run_instant_ngp(
    project,
):

    print()
    print("=" * 80)
    print("STARTING INSTANT-NGP")
    print("=" * 80)

    print(
        f"Executable: {INSTANT_NGP}"
    )

    print(
        f"Scene:      {project}"
    )

    subprocess.run(
        [
            str(INSTANT_NGP),
            "--scene",
            str(project),
        ],
        cwd=str(ROOT),
        check=True,
    )


# =============================================================================
# MAIN
# =============================================================================

def main():

    args = parse_args()

    # =========================================================================
    # ARGUMENT VALIDATION
    # =========================================================================

    if not args.project.strip():
        raise RuntimeError(
            "Project name cannot be empty."
        )

    if not args.prompt.strip():
        raise RuntimeError(
            "SAM 3 prompt cannot be empty."
        )

    if (
        args.frames is not None
        and args.frames < 1
    ):
        raise RuntimeError(
            "Frame count must be a positive integer."
        )

    # =========================================================================
    # PROJECT
    # =========================================================================

    project = (
        TESTDATA /
        args.project
    )

    if not project.exists():
        raise RuntimeError(
            f"Project directory does not exist:\n{project}"
        )

    if not project.is_dir():
        raise RuntimeError(
            f"Project path is not a directory:\n{project}"
        )

    # =========================================================================
    # EXTERNAL TOOLS
    # =========================================================================

    colmap = find_colmap(
        args.colmap
    )

    if not args.sam3.exists():
        raise RuntimeError(
            f"SAM 3 model does not exist:\n{args.sam3}"
        )

    if not INSTANT_NGP.exists():
        raise RuntimeError(
            f"Instant-NGP executable does not exist:\n{INSTANT_NGP}"
        )

    # =========================================================================
    # DETECT CAMERA SETUP AND SOURCE VIDEOS
    # =========================================================================

    (
        camera_count,
        active_cameras,
        camera_pairs,
        source_videos,
    ) = detect_camera_setup(
        project
    )

    # =========================================================================
    # DISPLAY CONFIGURATION
    # =========================================================================

    print()
    print("=" * 80)

    print(
        f"DETECTED {camera_count}-CAMERA SETUP"
    )

    print("=" * 80)
    print()

    print(
        "Source videos:"
    )

    for camera in active_cameras:

        print(
            f"  {camera:15s} -> "
            f"{source_videos[camera].name}"
        )

    print()

    print(
        "COLMAP:"
    )

    print(
        f"  {colmap}"
    )

    print()

    print(
        "SAM 3 model:"
    )

    print(
        f"  {args.sam3}"
    )

    print()

    print(
        "Matching topology:"
    )

    for camera_a, camera_b in camera_pairs:

        print(
            f"  {camera_a} <-> {camera_b}"
        )

    # =========================================================================
    # FRESH START
    # =========================================================================

    clean_project(
        project,
        source_videos,
    )

    # =========================================================================
    # PHASE 1: SAM 3 -> RGBA
    # =========================================================================

    run_phase_1(
        project=project,
        active_cameras=active_cameras,
        source_videos=source_videos,
        frame_count=args.frames,
        prompt=args.prompt,
        sam3_path=args.sam3,
    )

    # =========================================================================
    # PHASE 2: COLMAP
    # =========================================================================

    image_dir = (
        project /
        "images"
    )

    database = (
        project /
        "database.db"
    )

    sparse_dir = (
        project /
        "sparse"
    )

    text_dir = (
        project /
        "colmap_text"
    )

    camera_images = load_camera_images(
        project,
        active_cameras,
    )

    validate_rgba_images(
        camera_images
    )

    pair_file = create_image_pairs(
        project,
        camera_images,
        camera_pairs,
    )

    # =========================================================================
    # FEATURE EXTRACTION
    # =========================================================================

    run_feature_extraction(
        colmap,
        image_dir,
        database,
        args.feature,
    )

    # =========================================================================
    # FEATURE MATCHING
    # =========================================================================

    run_feature_matching(
        colmap,
        database,
        pair_file,
        args.feature,
        args.matcher,
    )

    # =========================================================================
    # MAPPING
    # =========================================================================

    run_mapper(
        colmap,
        image_dir,
        database,
        sparse_dir,
    )

    # =========================================================================
    # SELECT BEST MODEL
    # =========================================================================

    (
        best_model,
        best_text,
        best_count,
    ) = find_best_model(
        colmap,
        sparse_dir,
        text_dir,
    )

    # =========================================================================
    # TRANSFORMS.JSON
    # =========================================================================

    transforms = create_transforms(
        project,
        best_text,
    )

    transforms_path = (
        project /
        "transforms.json"
    )

    with open(
        transforms_path,
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            transforms,
            f,
            indent=2,
        )

    # =========================================================================
    # FINAL REPORT
    # =========================================================================

    print()
    print("=" * 80)
    print("COLMAP PHASE COMPLETE")
    print("=" * 80)
    print()

    print(
        f"Camera setup:      {camera_count}"
    )

    print(
        f"Feature extractor: "
        f"{'ALIKED-N32' if args.feature == 'aliked' else 'SIFT'}"
    )

    print(
        f"Feature matcher:   "
        f"{get_matcher_type(args.feature, args.matcher)}"
    )

    print(
        f"Best model:        {best_model.name}"
    )

    print(
        f"Registered images: {best_count}"
    )

    print(
        f"Transform frames:  "
        f"{len(transforms['frames'])}"
    )

    print(
        f"transforms.json:   {transforms_path}"
    )

    # =========================================================================
    # INSTANT-NGP
    # =========================================================================

    run_instant_ngp(
        project
    )


# =============================================================================
# PHASE 1
# =============================================================================

def run_phase_1(
    project,
    active_cameras,
    source_videos,
    frame_count,
    prompt,
    sam3_path,
):

    from ultralytics.models.sam import (
        SAM3SemanticPredictor
    )

    image_dir = (
        project /
        "images"
    )

    image_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    print()
    print("=" * 80)
    print("PHASE 1: SAM 3 -> RGBA")
    print("=" * 80)

    if frame_count is None:

        print(
            "Frame selection: approximately 2 FPS"
        )

    else:

        print(
            f"Frame selection: "
            f"{frame_count} frames per video"
        )

    print(
        f"SAM 3 model: {sam3_path}"
    )

    print(
        f"SAM 3 prompt: {prompt}"
    )

    print()

    print(
        "Loading SAM 3..."
    )

    predictor = SAM3SemanticPredictor(
        overrides={
            "model": str(sam3_path),
            "task": "segment",
            "mode": "predict",
            "conf": SAM3_CONF,
            "quantize": SAM3_QUANTIZE,
            "verbose": False,
            "save": False,
        }
    )

    print(
        "SAM 3 loaded."
    )

    for camera in active_cameras:

        output_dir = (
            image_dir /
            camera
        )

        create_rgba_from_video(
            video=source_videos[camera],
            output_dir=output_dir,
            count=frame_count,
            predictor=predictor,
            prompt=prompt,
        )

    print()
    print("=" * 80)
    print("PHASE 1 COMPLETE")
    print("=" * 80)


# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == "__main__":

    try:

        main()

    except subprocess.CalledProcessError as exc:

        print()
        print("=" * 80)
        print("EXTERNAL PROGRAM FAILED")
        print("=" * 80)
        print()

        print(
            f"Exit code: {exc.returncode}"
        )

        sys.exit(
            exc.returncode
        )

    except KeyboardInterrupt:

        print()

        print(
            "Pipeline interrupted by user."
        )

        sys.exit(130)

    except Exception as exc:

        print()
        print("=" * 80)
        print("PIPELINE ERROR")
        print("=" * 80)
        print()

        print(
            str(exc)
        )

        sys.exit(1)
