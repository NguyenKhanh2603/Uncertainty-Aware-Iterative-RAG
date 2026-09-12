from scripts.prepare_official_conformal_bundle import (
    QuestionLimiter,
    _existing_webqa_image_paths,
)


def test_question_limiter_caps_train_and_holdout_independently():
    limiter = QuestionLimiter(per_role_limit=2)

    assert [limiter.accept("train") for _ in range(3)] == [True, True, False]
    assert [limiter.accept("heldout") for _ in range(3)] == [True, True, False]
    assert limiter.counts == {"train": 2, "heldout": 2}


def test_existing_webqa_images_are_indexed_in_one_pass(tmp_path):
    destination = tmp_path / "webqa" / "images"
    destination.mkdir(parents=True)
    (destination / "11.jpg").write_bytes(b"image")
    (destination / "22.png").write_bytes(b"image")
    (destination / "unselected.jpg").write_bytes(b"image")

    assert _existing_webqa_image_paths(destination, {"11", "22", "missing"}) == {
        "11": "webqa/images/11.jpg",
        "22": "webqa/images/22.png",
    }
