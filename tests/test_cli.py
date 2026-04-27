"""BDD tests for the Typer CLI."""

from typer.testing import CliRunner

from sakura_simulator.cli import app

runner = CliRunner()


class TestHelloCommand:
    def test_given_hello_command_when_invoked_then_exits_successfully(self):
        # Given: the CLI app
        # When: `sakura hello` is invoked
        result = runner.invoke(app, ["hello"])
        # Then: exit code is 0
        assert result.exit_code == 0

    def test_given_hello_command_when_invoked_then_prints_full_greeting(self):
        # Given: the CLI app
        # When: `sakura hello` is invoked
        result = runner.invoke(app, ["hello"])
        # Then: the exact greeting appears in stdout
        assert "Hello from Sakura-II: Titan Biosignature Engine Active" in result.output
