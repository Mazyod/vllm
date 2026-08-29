# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""One command, from no machine to results on disk and nothing left running.

The ordering is the whole point: a run that destroys the box before fetching
what it measured has spent the money and thrown away the answer.
"""

import json

import pytest

from fork.bench.__main__ import main
from fork.bench.campaign import run_campaign
from fork.bench.provision import Instance, NewInstance, Offer
from fork.bench.remote import DEFAULT_SSH_DEADLINE_S

OFFER = Offer(
    id=7,
    gpu_name="H100_PCIE",
    num_gpus=2,
    dph=3.0,
    disk_gb=400.0,
    verified=True,
    rentable=True,
    direct_port_count=2,
)


class FakeProvider:
    """A provider that records what happened to it, in order.

    Attributes:
        events: Lifecycle events, in the order they occurred.
        labels: Labels every created instance carried.
        specs: Everything it was asked to put on a machine.
        boot_status: Status the created instance reports from then on.
    """

    def __init__(self, boot_status="running"):
        self.events: list[str] = []
        self.labels: list[str] = []
        self.specs: list = []
        self.live: dict[int, Instance] = {}
        self.boot_status = boot_status

    def search_offers(self, requirements):
        return [OFFER]

    def create(self, offer, spec):
        self.events.append("create")
        self.labels.append(spec.label)
        self.specs.append(spec)
        self.live[100] = Instance(id=100, status=self.boot_status, label=spec.label)
        return NewInstance(id=100, key="scoped-secret")

    def describe(self, instance_id):
        return self.live.get(instance_id)

    def instances(self):
        return list(self.live.values())

    def destroy(self, instance_id):
        self.events.append("destroy")
        self.live.pop(instance_id, None)

    def ssh_endpoint(self, instance_id):
        return "ssh5.example.net", 41234


class FakeShell:
    """Runs nothing; records what it was asked to run.

    Attributes:
        events: One event per invocation, named by the kind of command.
    """

    def __init__(self, exit_code="0", fail_on=()):
        self.events: list[str] = []
        self.commands: list[list[str]] = []
        self.exit_code = exit_code
        self.fail_on = {fail_on} if isinstance(fail_on, str) else set(fail_on)

    def __call__(self, argv):
        self.commands.append(list(argv))
        joined = " ".join(argv)
        if argv[0] == "rsync":
            kind = "collect" if ":" in argv[-2] else "push"
        elif "nohup" in joined:
            kind = "start"
        elif "gate.exit" in joined:
            kind = "poll"
        elif "mkdir" in joined:
            kind = "mkdir"
        else:
            kind = "ready"
        self.events.append(kind)
        if kind in self.fail_on:
            # run_argv reports a failure by quoting the whole command, which
            # is what makes redacting one a thing worth testing.
            raise RuntimeError(f"{joined} exited 255: {kind} failed")
        return self.exit_code if kind == "poll" else ""


def _campaign(provider, shell, out_dir, **kwargs):
    return run_campaign(
        tag="v0.27.1",
        image="img:tag",
        out_dir=out_dir,
        provider=provider,
        shell=shell,
        phases=(4,),
        poll_s=0,
        settle_s=0,
        **kwargs,
    )


def test_a_campaign_returns_the_exit_code_the_gate_reached(tmp_path):
    assert _campaign(FakeProvider(), FakeShell("0"), tmp_path) == 0
    assert _campaign(FakeProvider(), FakeShell("1"), tmp_path) == 1


def test_a_campaign_fetches_results_before_it_destroys_the_box(tmp_path):
    """Destroying first spends the money and discards the answer."""
    provider, shell = FakeProvider(), FakeShell("0")
    _campaign(provider, shell, tmp_path)
    assert "collect" in shell.events
    assert provider.events == ["create", "destroy"]


def test_a_campaign_makes_the_workdir_before_pushing_into_it(tmp_path):
    """rsync creates the last path component, not the ones above it."""
    shell = FakeShell("0")
    _campaign(FakeProvider(), shell, tmp_path)
    assert shell.events.index("mkdir") < shell.events.index("push")


def test_a_campaign_fetches_results_when_the_gate_failed(tmp_path):
    """A failing run's logs are the ones worth having."""
    provider, shell = FakeProvider(), FakeShell("1")
    _campaign(provider, shell, tmp_path)
    assert "collect" in shell.events


def test_a_campaign_fetches_results_when_the_gate_never_finished(tmp_path):
    provider, shell = FakeProvider(), FakeShell("")
    code = _campaign(provider, shell, tmp_path, gate_deadline_s=0)
    assert code != 0
    assert "collect" in shell.events
    assert "destroy" in provider.events


def test_a_campaign_destroys_the_box_when_the_push_fails(tmp_path):
    provider = FakeProvider()
    with pytest.raises(RuntimeError):
        _campaign(provider, FakeShell(fail_on="push"), tmp_path)
    assert "destroy" in provider.events


def test_a_campaign_labels_the_instance_with_the_release_under_test(tmp_path):
    """The label is what lets a stray be found and destroyed later."""
    provider = FakeProvider()
    _campaign(provider, FakeShell("0"), tmp_path)
    assert provider.labels == ["fork-bench-v0.27.1"]


def test_a_campaign_never_writes_the_instance_key_to_disk(tmp_path):
    """The key can destroy the instance; it belongs in memory only."""
    _campaign(FakeProvider(), FakeShell("0"), tmp_path)
    written = "\n".join(
        path.read_text(encoding="utf-8")
        for path in tmp_path.rglob("*")
        if path.is_file()
    )
    assert "scoped-secret" not in written


def test_a_campaign_records_what_it_rented(tmp_path):
    """A number without the machine behind it cannot be compared to anything."""
    _campaign(FakeProvider(), FakeShell("0"), tmp_path)
    rental = (tmp_path / "rental.json").read_text(encoding="utf-8")
    assert "H100_PCIE" in rental
    assert "3.0" in rental


def test_a_campaign_records_the_instance_id_before_the_box_is_waited_on(tmp_path):
    """A driver killed during boot must still leave a backstop something to read."""
    provider = FakeProvider(boot_status="error")
    with pytest.raises(RuntimeError):
        _campaign(provider, FakeShell("0"), tmp_path)
    record = json.loads((tmp_path / "rental.json").read_text(encoding="utf-8"))
    assert record["instance_id"] == 100
    assert record["label"] == "fork-bench-v0.27.1"


def test_a_campaign_waits_for_ssh_before_it_uses_it(tmp_path):
    """A box the provider calls running may still refuse the first login."""
    shell = FakeShell("0")
    _campaign(FakeProvider(), shell, tmp_path)
    assert shell.events.index("ready") < shell.events.index("mkdir")


def test_a_failing_collect_does_not_replace_the_reason_the_run_failed(tmp_path):
    """An rsync error standing in front of the real cause hid two of them."""
    provider = FakeProvider()
    with pytest.raises(RuntimeError, match="push failed"):
        _campaign(provider, FakeShell(fail_on=("push", "collect")), tmp_path)
    assert "destroy" in provider.events
    assert "collect failed" in (tmp_path / "collect-failed.txt").read_text(
        encoding="utf-8"
    )


def test_a_run_that_could_not_be_collected_is_still_loud(tmp_path):
    """Results left on a box that is about to be destroyed are not results."""
    with pytest.raises(RuntimeError, match="collect failed"):
        _campaign(FakeProvider(), FakeShell("0", fail_on="collect"), tmp_path)


def test_a_collect_that_worked_clears_the_last_attempts_marker(tmp_path):
    """Run directories are reused; a stale marker describes results that are here."""
    with pytest.raises(RuntimeError):
        _campaign(FakeProvider(), FakeShell("0", fail_on="collect"), tmp_path)
    _campaign(FakeProvider(), FakeShell("0"), tmp_path)
    assert not (tmp_path / "collect-failed.txt").exists()


def test_a_campaign_gives_the_provider_no_environment_to_pass_on(tmp_path):
    """That argument declares the port mappings; one without them drops ssh."""
    provider = FakeProvider()
    _campaign(provider, FakeShell("0"), tmp_path, env={"HF_TOKEN": "shh"})
    assert "shh" not in repr(provider.specs[0])


def test_a_campaign_hands_the_token_to_the_gate_over_ssh(tmp_path):
    """The gate still needs it to download a gated checkpoint."""
    shell = FakeShell("0")
    _campaign(FakeProvider(), shell, tmp_path, env={"HF_TOKEN": "shh"})
    start = next(command for command in shell.commands if "nohup" in " ".join(command))
    assert "export HF_TOKEN=shh" in " ".join(start)


def test_a_gate_that_will_not_start_does_not_quote_the_token(tmp_path):
    """The failure report is the whole command, and the command holds a secret."""
    with pytest.raises(RuntimeError) as raised:
        _campaign(
            FakeProvider(),
            FakeShell("0", fail_on="start"),
            tmp_path,
            env={"HF_TOKEN": "shh"},
        )
    assert "shh" not in str(raised.value)
    assert "HF_TOKEN" in str(raised.value)


def test_a_token_needing_quotes_is_redacted_too(tmp_path):
    """Quoted into the command, the raw value never appears to be matched."""
    with pytest.raises(RuntimeError) as raised:
        _campaign(
            FakeProvider(),
            FakeShell("0", fail_on="start"),
            tmp_path,
            env={"HF_TOKEN": "hf_ab'cd"},
        )
    assert "hf_ab" not in str(raised.value)


def test_an_interrupt_during_collection_does_not_hide_the_run_failure(tmp_path):
    """The narrower catch let exactly this reintroduce the defect."""

    class Interrupting(FakeShell):
        def __call__(self, argv):
            if argv[0] == "rsync" and ":" in argv[-2]:
                raise KeyboardInterrupt
            return super().__call__(argv)

    with pytest.raises(RuntimeError, match="push failed"):
        _campaign(FakeProvider(), Interrupting("0", fail_on="push"), tmp_path)


def test_a_campaign_passes_the_boot_image_to_the_on_box_gate(tmp_path):
    shell = FakeShell("0")
    _campaign(FakeProvider(), shell, tmp_path)
    start = next(
        command for command in shell.commands if "gate.exit" in " ".join(command)
    )
    assert "--image img:tag" in " ".join(start)


def test_renting_from_the_command_line_needs_an_image(tmp_path):
    """Nothing to boot means nothing to test; fail before the meter starts."""
    with pytest.raises(SystemExit):
        main(["--tag", "v0.27.1", "--out", str(tmp_path), "--rent"])


def test_renting_from_the_command_line_passes_the_run_through(tmp_path, monkeypatch):
    captured = {}

    def fake_campaign(**kwargs):
        captured.update(kwargs)
        return 0

    monkeypatch.setattr("fork.bench.campaign.run_campaign", fake_campaign)
    main(
        [
            "--tag",
            "v0.27.1",
            "--image",
            "img:tag",
            "--out",
            str(tmp_path),
            "--phase",
            "4",
            "--rent",
        ]
    )
    assert captured["tag"] == "v0.27.1"
    assert captured["image"] == "img:tag"
    assert tuple(captured["phases"]) == (4,)


def test_the_cap_bounds_what_a_run_can_cost(tmp_path, monkeypatch):
    captured = {}
    monkeypatch.setattr(
        "fork.bench.campaign.run_campaign", lambda **kw: captured.update(kw) or 0
    )
    main(
        [
            "--tag",
            "v0.27.1",
            "--image",
            "i",
            "--out",
            str(tmp_path),
            "--rent",
            "--cap-minutes",
            "150",
        ]
    )
    assert captured["cap_seconds"] == 150 * 60


def test_the_gate_gives_up_before_the_reaper_fires(tmp_path, monkeypatch):
    """Otherwise the box is destroyed out from under the results."""
    captured = {}
    monkeypatch.setattr(
        "fork.bench.campaign.run_campaign", lambda **kw: captured.update(kw) or 0
    )
    main(
        [
            "--tag",
            "v0.27.1",
            "--image",
            "i",
            "--out",
            str(tmp_path),
            "--rent",
            "--cap-minutes",
            "150",
        ]
    )
    assert captured["gate_deadline_s"] < captured["cap_seconds"]


def test_a_cap_too_short_to_collect_results_is_refused(tmp_path):
    with pytest.raises(SystemExit):
        main(
            [
                "--tag",
                "v0.27.1",
                "--image",
                "i",
                "--out",
                str(tmp_path),
                "--rent",
                "--cap-minutes",
                "5",
            ]
        )


def test_the_login_budget_can_be_raised_for_a_slow_venue(tmp_path, monkeypatch):
    """Five minutes was not enough on real hardware; the number is an argument."""
    captured = {}
    monkeypatch.setattr(
        "fork.bench.campaign.run_campaign", lambda **kw: captured.update(kw) or 0
    )
    main(
        [
            "--tag",
            "v0.27.1",
            "--image",
            "i",
            "--out",
            str(tmp_path),
            "--rent",
            "--ssh-deadline-minutes",
            "30",
        ]
    )
    assert captured["ssh_deadline_s"] == 30 * 60


def test_the_login_budget_defaults_to_the_one_the_harness_justifies(
    tmp_path, monkeypatch
):
    captured = {}
    monkeypatch.setattr(
        "fork.bench.campaign.run_campaign", lambda **kw: captured.update(kw) or 0
    )
    main(["--tag", "v0.27.1", "--image", "i", "--out", str(tmp_path), "--rent"])
    assert captured["ssh_deadline_s"] == DEFAULT_SSH_DEADLINE_S


def test_renting_hands_the_box_the_token_it_needs(tmp_path, monkeypatch):
    """Gated checkpoints do not download without it, and the run wastes a box."""
    captured = {}
    monkeypatch.setattr(
        "fork.bench.campaign.run_campaign", lambda **kw: captured.update(kw) or 0
    )
    monkeypatch.setenv("HF_TOKEN", "from-the-environment")
    main(["--tag", "v0.27.1", "--image", "i", "--out", str(tmp_path), "--rent"])
    assert captured["env"]["HF_TOKEN"] == "from-the-environment"
