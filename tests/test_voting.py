from apps.inference_worker.voting import Candidate, TrackVoter


def c(text: str, province: str | None = "PP", conf: float = 0.9) -> Candidate:
    return Candidate(plate_text=text, province_code=province, confidence=conf)


def test_commits_after_quorum():
    voter = TrackVoter(votes_required=3)
    assert voter.add("t1", c("2D-0888")) is None
    assert voter.add("t1", c("2D-0888")) is None
    result = voter.add("t1", c("2D-0888"))
    assert result is not None
    assert result.plate_text == "2D-0888"
    assert result.votes == 3


def test_outlier_frame_does_not_win():
    voter = TrackVoter(votes_required=3)
    voter.add("t1", c("2D-0888"))
    voter.add("t1", c("2D-088B"))
    voter.add("t1", c("2D-0888"))
    result = voter.add("t1", c("2D-0888"))
    assert result is not None
    assert result.plate_text == "2D-0888"
    assert result.total == 4


def test_track_commits_only_once():
    voter = TrackVoter(votes_required=2)
    voter.add("t1", c("2D-0888"))
    assert voter.add("t1", c("2D-0888")) is not None
    assert voter.add("t1", c("2D-0888")) is None


def test_tracks_are_independent():
    voter = TrackVoter(votes_required=2)
    voter.add("t1", c("2D-0888"))
    voter.add("t2", c("3A-1111"))
    assert voter.add("t1", c("2D-0888")).plate_text == "2D-0888"
    assert voter.add("t2", c("3A-1111")).plate_text == "3A-1111"


def test_close_resets_track():
    voter = TrackVoter(votes_required=2)
    voter.add("t1", c("2D-0888"))
    voter.close("t1")
    assert voter.add("t1", c("2D-0888")) is None
