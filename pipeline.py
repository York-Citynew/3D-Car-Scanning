import argparse
import cv2
import json
import numpy as np
from pathlib import Path
import shutil
import subprocess

from ultralytics.models.sam import SAM3SemanticPredictor


def run(colmap, *args):
    subprocess.run(
        [str(colmap), *map(str, args)],
        check=True
    )


def video_info(video):
    cap = cv2.VideoCapture(str(video))

    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    cap.release()

    if fps <= 0 or total_frames <= 0:
        raise RuntimeError(
            "Could not determine video FPS/frame count"
        )

    duration = total_frames / fps
    default_frames = max(1, round(duration))

    return fps, total_frames, duration, default_frames


def get_frame_indices(total_frames, count):
    if count < 1:
        raise RuntimeError(
            "Frame count must be at least 1"
        )

    if count > total_frames:
        raise RuntimeError(
            f"Requested {count} frames, "
            f"but video contains only {total_frames} frames"
        )

    if count == 1:
        return np.array([0], dtype=np.int64)

    return np.linspace(
        0,
        total_frames - 1,
        count
    ).round().astype(np.int64)


def clean_project(project):
    for name in [
        "images",
        "rgba",
        "sparse",
        "colmap_text",
    ]:
        shutil.rmtree(
            project / name,
            ignore_errors=True
        )

    (project / "database.db").unlink(
        missing_ok=True
    )

    (project / "transforms.json").unlink(
        missing_ok=True
    )


def create_original_images(
    video,
    image_dir,
    count
):
    image_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    _, total_frames, _, _ = video_info(video)

    indices = get_frame_indices(
        total_frames,
        count
    )

    wanted = {
        int(frame_index): index + 1
        for index, frame_index in enumerate(indices)
    }

    cap = cv2.VideoCapture(str(video))

    if not cap.isOpened():
        raise RuntimeError(
            f"Could not open video: {video}"
        )

    frame_idx = 0
    written = 0

    while True:
        ret, frame = cap.read()

        if not ret:
            break

        if frame_idx in wanted:
            out_idx = wanted[frame_idx]

            output = (
                image_dir /
                f"{out_idx:04d}.jpg"
            )

            if not cv2.imwrite(
                str(output),
                frame
            ):
                raise RuntimeError(
                    f"Could not write {output}"
                )

            written += 1

            if written == count:
                break

        frame_idx += 1

    cap.release()

    if written != count:
        raise RuntimeError(
            f"Expected {count} images, "
            f"created {written}"
        )


def create_combined_mask(
    results,
    frame
):
    if not (
        results
        and results[0].masks is not None
        and len(results[0].masks.data)
    ):
        return None, 0

    masks = (
        results[0]
        .masks
        .data
        .cpu()
        .numpy()
    )

    h, w = frame.shape[:2]

    combined_mask = np.zeros(
        (h, w),
        dtype=np.uint8
    )

    for mask in masks:
        if mask.shape != (h, w):
            mask = cv2.resize(
                mask,
                (w, h),
                interpolation=cv2.INTER_NEAREST
            )

        combined_mask = np.maximum(
            combined_mask,
            (mask > 0).astype(np.uint8) * 255
        )

    return combined_mask, len(masks)


def create_rgba_from_video(
    video,
    out_dir,
    count,
    prompt,
    sam3_path
):
    out_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    _, total_frames, _, _ = video_info(video)

    indices = get_frame_indices(
        total_frames,
        count
    )

    wanted = {
        int(frame_index): index + 1
        for index, frame_index in enumerate(indices)
    }

    predictor = SAM3SemanticPredictor(
        overrides={
            "model": str(sam3_path),
            "task": "segment",
            "mode": "predict",
            "conf": 0.25,
            "quantize": 16,
            "verbose": False,
            "save": False,
        }
    )

    cap = cv2.VideoCapture(str(video))

    if not cap.isOpened():
        raise RuntimeError(
            f"Could not open video: {video}"
        )

    frame_idx = 0
    written = 0

    while True:
        ret, frame = cap.read()

        if not ret:
            break

        if frame_idx in wanted:
            out_idx = wanted[frame_idx]

            predictor.set_image(frame)

            results = predictor(
                text=[prompt]
            )

            alpha, object_count = create_combined_mask(
                results,
                frame
            )

            if alpha is not None:
                rgba = np.dstack(
                    (frame, alpha)
                )

                output = (
                    out_dir /
                    f"{out_idx:04d}.png"
                )

                if not cv2.imwrite(
                    str(output),
                    rgba
                ):
                    raise RuntimeError(
                        f"Could not write {output}"
                    )

                written += 1

                print(
                    f"Frame {out_idx:04d}: "
                    f"{object_count} matching objects"
                )

            else:
                print(
                    f"WARNING: No mask found "
                    f"for frame {out_idx:04d}"
                )

        frame_idx += 1

        if frame_idx > indices[-1]:
            break

    cap.release()

    if written == 0:
        raise RuntimeError(
            "SAM3 did not produce any RGBA images"
        )

    print(
        f"Created {written} RGBA images"
    )


def create_rgba_from_images(
    image_dir,
    out_dir,
    prompt,
    sam3_path
):
    out_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    predictor = SAM3SemanticPredictor(
        overrides={
            "model": str(sam3_path),
            "task": "segment",
            "mode": "predict",
            "conf": 0.25,
            "quantize": 16,
            "verbose": False,
            "save": False,
        }
    )

    image_paths = sorted(
        image_dir.glob("*.jpg")
    )

    written = 0

    for image_path in image_paths:
        frame = cv2.imread(
            str(image_path)
        )

        if frame is None:
            raise RuntimeError(
                f"Could not read {image_path}"
            )

        predictor.set_image(frame)

        results = predictor(
            text=[prompt]
        )

        alpha, object_count = create_combined_mask(
            results,
            frame
        )

        if alpha is not None:
            rgba = np.dstack(
                (frame, alpha)
            )

            output = (
                out_dir /
                f"{image_path.stem}.png"
            )

            if not cv2.imwrite(
                str(output),
                rgba
            ):
                raise RuntimeError(
                    f"Could not write {output}"
                )

            written += 1

            print(
                f"{image_path.name}: "
                f"{object_count} matching objects"
            )

        else:
            print(
                f"WARNING: No mask found "
                f"for {image_path.name}"
            )

    if written == 0:
        raise RuntimeError(
            "SAM3 did not produce any RGBA images"
        )

    print(
        f"Created {written} RGBA images"
    )


def run_colmap(
    colmap,
    image_dir,
    db,
    sparse,
    feature_type,
    match_type,
    guided,
    match_mode,
    overlap,
    quadratic
):
    if feature_type == "1":
        extraction = "ALIKED_N32"
    else:
        extraction = "SIFT"

    if feature_type == "1":
        if match_type == "1":
            matcher = "ALIKED_LIGHTGLUE"
        else:
            matcher = "ALIKED_BRUTEFORCE"
    else:
        if match_type == "1":
            matcher = "SIFT_LIGHTGLUE"
        else:
            matcher = "SIFT_BRUTEFORCE"

    if extraction == "ALIKED_N32":
        feature_args = [
            "--AlikedExtraction.max_num_features",
            "2048",
            "--AlikedExtraction.min_score",
            "0.2",
        ]
    else:
        feature_args = [
            "--SiftExtraction.max_num_features",
            "8192",
        ]

    run(
        colmap,
        "feature_extractor",
        "--database_path",
        db,
        "--image_path",
        image_dir,
        "--ImageReader.single_camera",
        "1",
        "--ImageReader.camera_model",
        "OPENCV",
        "--FeatureExtraction.type",
        extraction,
        "--FeatureExtraction.use_gpu",
        "1",
        *feature_args,
    )

    if match_mode == "1":
        match_args = [
            "sequential_matcher",
            "--database_path",
            db,
            "--FeatureMatching.type",
            matcher,
            "--FeatureMatching.use_gpu",
            "1",
            "--SequentialMatching.overlap",
            overlap,
            "--SequentialMatching.quadratic_overlap",
            quadratic,
            "--SequentialMatching.loop_detection",
            "0",
        ]
    else:
        match_args = [
            "exhaustive_matcher",
            "--database_path",
            db,
            "--FeatureMatching.type",
            matcher,
            "--FeatureMatching.use_gpu",
            "1",
        ]

    if match_type == "2":
        match_args.extend([
            "--FeatureMatching.guided_matching",
            guided,
        ])

    run(
        colmap,
        *match_args
    )

    sparse.mkdir(
        parents=True,
        exist_ok=True
    )

    run(
        colmap,
        "mapper",
        "--database_path",
        db,
        "--image_path",
        image_dir,
        "--output_path",
        sparse,
        "--Mapper.multiple_models",
        "1",
        "--Mapper.structure_less_registration_fallback",
        "1",
        "--Mapper.min_num_matches",
        "15",
        "--Mapper.init_min_num_inliers",
        "100",
        "--Mapper.abs_pose_min_num_inliers",
        "30",
        "--Mapper.abs_pose_min_inlier_ratio",
        "0.25",
        "--Mapper.init_max_error",
        "4",
        "--Mapper.filter_max_reproj_error",
        "4",
        "--Mapper.init_min_tri_angle",
        "16",
    )


def find_best_model(
    colmap,
    sparse,
    text
):
    models = []

    if not sparse.exists():
        raise RuntimeError(
            "COLMAP sparse directory was not created"
        )

    for model_dir in sorted(
        sparse.iterdir()
    ):
        if not model_dir.is_dir():
            continue

        if not (
            model_dir /
            "images.bin"
        ).exists():
            continue

        model_text = (
            text /
            model_dir.name
        )

        if model_text.exists():
            shutil.rmtree(
                model_text
            )

        model_text.mkdir(
            parents=True,
            exist_ok=True
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

        with open(
            images_txt,
            encoding="utf-8"
        ) as f:
            lines = f.readlines()

        count = 0
        i = 0

        while i < len(lines):
            line = lines[i].strip()

            if not line or line.startswith("#"):
                i += 1
                continue

            e = line.split()

            if len(e) >= 10:
                count += 1

            i += 2

        models.append(
            (
                model_dir,
                model_text,
                count,
            )
        )

    if not models:
        raise RuntimeError(
            "COLMAP created no usable reconstruction model"
        )

    models.sort(
        key=lambda x: x[2],
        reverse=True
    )

    print()
    print("COLMAP models:")

    for model_dir, _, count in models:
        print(
            f"  model {model_dir.name}: "
            f"{count} registered images"
        )

    best_model, best_text, best_count = (
        models[0]
    )

    print(
        f"Using best model: "
        f"{best_model.name} "
        f"with {best_count} registered images"
    )

    return (
        best_model,
        best_text,
        best_count
    )


def create_transforms(
    best_text,
    rgba_dir
):
    with open(
        best_text / "cameras.txt",
        encoding="utf-8"
    ) as f:
        for line in f:
            line = line.strip()

            if not line or line.startswith("#"):
                continue

            e = line.split()

            if e[1] != "OPENCV":
                raise RuntimeError(
                    f"Expected OPENCV, got {e[1]}"
                )

            w, h = map(
                float,
                e[2:4]
            )

            fx, fy = map(
                float,
                e[4:6]
            )

            cx, cy = map(
                float,
                e[6:8]
            )

            k1, k2, p1, p2 = map(
                float,
                e[8:12]
            )

            camera = {
                "w": w,
                "h": h,
                "fl_x": fx,
                "fl_y": fy,
                "k1": k1,
                "k2": k2,
                "k3": 0,
                "k4": 0,
                "p1": p1,
                "p2": p2,
                "cx": cx,
                "cy": cy,
                "is_fisheye": False,
                "camera_angle_x": (
                    2 *
                    np.arctan(
                        w /
                        (2 * fx)
                    )
                ),
                "camera_angle_y": (
                    2 *
                    np.arctan(
                        h /
                        (2 * fy)
                    )
                ),
            }

            break

        else:
            raise RuntimeError(
                "No camera found"
            )

    frames = []
    up = np.zeros(
        3,
        dtype=np.float64
    )

    with open(
        best_text / "images.txt",
        encoding="utf-8"
    ) as f:
        lines = f.readlines()

    i = 0

    while i < len(lines):
        line = lines[i].strip()

        if not line or line.startswith("#"):
            i += 1
            continue

        e = line.split()

        if len(e) < 10:
            i += 1
            continue

        qvec = np.array([
            float(x)
            for x in e[1:5]
        ])

        tvec = np.array([
            float(x)
            for x in e[5:8]
        ])

        R = qvec2rotmat(-qvec)

        m = np.eye(4)
        m[:3, :3] = R
        m[:3, 3] = tvec

        c2w = np.linalg.inv(m)

        c2w[0:3, 2] *= -1
        c2w[0:3, 1] *= -1
        c2w = c2w[[1, 0, 2, 3]]
        c2w[2, :] *= -1

        up += c2w[0:3, 1]

        name = " ".join(e[9:])

        rgba_name = (
            Path(name).stem +
            ".png"
        )

        rgba_path = (
            rgba_dir /
            rgba_name
        )

        image = cv2.imread(
            str(rgba_path),
            cv2.IMREAD_UNCHANGED
        )

        sharpness = 0.0

        if image is not None:
            if (
                len(image.shape) == 3
                and image.shape[2] == 4
            ):
                gray = cv2.cvtColor(
                    image,
                    cv2.COLOR_BGRA2GRAY
                )
            else:
                gray = cv2.cvtColor(
                    image,
                    cv2.COLOR_BGR2GRAY
                )

            sharpness = cv2.Laplacian(
                gray,
                cv2.CV_64F
            ).var()

        frames.append({
            "file_path": (
                f"./rgba/{rgba_name}"
            ),
            "sharpness": float(
                sharpness
            ),
            "transform_matrix": c2w,
        })

        i += 2

    if not frames:
        raise RuntimeError(
            "No registered frames found in images.txt"
        )

    print(
        f"Registered frames: "
        f"{len(frames)}"
    )

    up /= np.linalg.norm(up)

    R = rotmat(
        up,
        np.array([
            0.0,
            0.0,
            1.0
        ])
    )

    R = np.pad(
        R,
        [(0, 1), (0, 1)]
    )

    R[-1, -1] = 1

    for frame in frames:
        frame["transform_matrix"] = (
            R @
            frame["transform_matrix"]
        )

    totw = 0.0
    totp = np.zeros(
        3,
        dtype=np.float64
    )

    for frame in frames:
        mf = (
            frame["transform_matrix"][:3]
        )

        for other in frames:
            mg = (
                other["transform_matrix"][:3]
            )

            p, w = closest_point_2_lines(
                mf[:, 3],
                mf[:, 2],
                mg[:, 3],
                mg[:, 2]
            )

            if w > 0.00001:
                totp += p * w
                totw += w

    if totw > 0:
        totp /= totw

    for frame in frames:
        frame["transform_matrix"][
            :3,
            3
        ] -= totp

    avglen = np.mean([
        np.linalg.norm(
            frame["transform_matrix"][
                :3,
                3
            ]
        )
        for frame in frames
    ])

    if avglen <= 0:
        raise RuntimeError(
            "Average camera distance is zero"
        )

    for frame in frames:
        frame["transform_matrix"][
            :3,
            3
        ] *= 4.0 / avglen

        frame["transform_matrix"] = (
            frame["transform_matrix"].tolist()
        )

    return {
        **camera,
        "aabb_scale": 4,
        "frames": frames,
    }


def qvec2rotmat(q):
    w, x, y, z = q

    return np.array([
        [
            1 - 2*y*y - 2*z*z,
            2*x*y - 2*w*z,
            2*x*z + 2*w*y,
        ],
        [
            2*x*y + 2*w*z,
            1 - 2*x*x - 2*z*z,
            2*y*z - 2*w*x,
        ],
        [
            2*x*z - 2*w*y,
            2*y*z + 2*w*x,
            1 - 2*x*x - 2*y*y,
        ],
    ])


def rotmat(a, b):
    a = np.asarray(
        a,
        dtype=np.float64
    )

    b = np.asarray(
        b,
        dtype=np.float64
    )

    a /= np.linalg.norm(a)
    b /= np.linalg.norm(b)

    v = np.cross(a, b)
    c = np.dot(a, b)

    if c < -1 + 1e-10:
        return rotmat(
            a +
            np.random.uniform(
                -1e-2,
                1e-2,
                3
            ),
            b
        )

    s = np.linalg.norm(v)

    k = np.array([
        [0.0, -v[2], v[1]],
        [v[2], 0.0, -v[0]],
        [-v[1], v[0], 0.0],
    ])

    return (
        np.eye(3) +
        k +
        k @ k *
        ((1 - c) /
         (s * s + 1e-10))
    )


def closest_point_2_lines(
    oa,
    da,
    ob,
    db
):
    da /= np.linalg.norm(da)
    db /= np.linalg.norm(db)

    c = oa - ob
    aa = np.dot(da, da)
    bb = np.dot(da, db)
    cc = np.dot(db, db)
    dd = np.dot(da, c)
    ee = np.dot(db, c)

    denom = aa * cc - bb * bb

    if abs(denom) < 1e-10:
        return (
            (oa + ob) * 0.5,
            0.0
        )

    ta = (
        bb * ee -
        cc * dd
    )

    ta /= denom

    tb = (
        aa * ee -
        bb * dd
    )

    tb /= denom

    p = (
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

    if distance <= 0:
        return p, 0.0

    w = 1.0 / distance

    return p, w


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--video-info",
        type=str
    )

    parser.add_argument(
        "--default-frames",
        type=str
    )

    parser.add_argument(
        "--project",
        type=Path
    )

    parser.add_argument(
        "--video",
        type=Path
    )

    parser.add_argument(
        "--frames",
        type=int
    )

    parser.add_argument(
        "--prompt",
        type=str
    )

    parser.add_argument(
        "--sam3",
        type=Path
    )

    parser.add_argument(
        "--colmap",
        type=Path
    )

    parser.add_argument(
        "--pipeline-mode",
        type=str
    )

    parser.add_argument(
        "--feature-type",
        type=str
    )

    parser.add_argument(
        "--match-type",
        type=str
    )

    parser.add_argument(
        "--guided",
        type=str,
        default="0"
    )

    parser.add_argument(
        "--match-mode",
        type=str
    )

    parser.add_argument(
        "--overlap",
        type=str,
        default="10"
    )

    parser.add_argument(
        "--quadratic",
        type=str,
        default="1"
    )

    args = parser.parse_args()

    if args.video_info:
        _, total_frames, duration, default_frames = (
            video_info(
                Path(args.video_info)
            )
        )

        fps = total_frames / duration

        print(
            f"Duration: {duration:.2f} seconds"
        )

        print(
            f"FPS: {fps:.3f}"
        )

        print(
            f"Total video frames: "
            f"{total_frames}"
        )

        print(
            f"Default 1 FPS frame count: "
            f"{default_frames}"
        )

        return

    if args.default_frames:
        default_frames = (
            video_info(
                Path(args.default_frames)
            )[3]
        )

        print(
            f"frames={default_frames}"
        )

        return

    if args.project is None:
        raise RuntimeError(
            "Project path is required"
        )

    if args.video is None:
        raise RuntimeError(
            "Video path is required"
        )

    if args.frames is None:
        raise RuntimeError(
            "Frame count is required"
        )

    if args.prompt is None:
        raise RuntimeError(
            "SAM3 prompt is required"
        )

    if args.sam3 is None:
        raise RuntimeError(
            "SAM3 model path is required"
        )

    if args.colmap is None:
        raise RuntimeError(
            "COLMAP path is required"
        )

    if not args.sam3.exists():
        raise RuntimeError(
            f"SAM3 model not found: {args.sam3}"
        )

    if not args.colmap.exists():
        raise RuntimeError(
            f"COLMAP not found: {args.colmap}"
        )

    if args.pipeline_mode not in (
        "1",
        "2",
    ):
        raise RuntimeError(
            "Invalid pipeline mode"
        )

    if args.feature_type not in (
        "1",
        "2",
    ):
        raise RuntimeError(
            "Invalid feature extraction type"
        )

    if args.match_type not in (
        "1",
        "2",
    ):
        raise RuntimeError(
            "Invalid matching type"
        )

    if args.match_mode not in (
        "1",
        "2",
    ):
        raise RuntimeError(
            "Invalid matching mode"
        )

    project = args.project

    clean_project(project)

    images_dir = (
        project /
        "images"
    )

    rgba_dir = (
        project /
        "rgba"
    )

    db = (
        project /
        "database.db"
    )

    sparse = (
        project /
        "sparse"
    )

    text = (
        project /
        "colmap_text"
    )

    if args.pipeline_mode == "1":
        create_rgba_from_video(
            args.video,
            rgba_dir,
            args.frames,
            args.prompt,
            args.sam3
        )

        colmap_images = rgba_dir

    else:
        create_original_images(
            args.video,
            images_dir,
            args.frames
        )

        colmap_images = images_dir

        create_rgba_from_images(
            images_dir,
            rgba_dir,
            args.prompt,
            args.sam3
        )

    print()
    print(
        "Running COLMAP..."
    )

    run_colmap(
        args.colmap,
        colmap_images,
        db,
        sparse,
        args.feature_type,
        args.match_type,
        args.guided,
        args.match_mode,
        args.overlap,
        args.quadratic,
    )

    (
        best_model,
        best_text,
        best_count,
    ) = find_best_model(
        args.colmap,
        sparse,
        text
    )

    transforms = create_transforms(
        best_text,
        rgba_dir
    )

    with open(
        project / "transforms.json",
        "w",
        encoding="utf-8"
    ) as f:
        json.dump(
            transforms,
            f,
            indent=2
        )

    print()
    print(
        f"Created transforms.json with "
        f"{len(transforms['frames'])} frames"
    )

    print(
        f"Best COLMAP model: "
        f"{best_model.name}"
    )

    print(
        f"Best model registered images: "
        f"{best_count}"
    )

    print(
        f"Project: {project}"
    )


if __name__ == "__main__":
    main()
