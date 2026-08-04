import unittest

from scripts.configure_openclaw import configure


class ConfigureOpenClawTest(unittest.TestCase):
    def test_preserves_unrelated_config_and_sets_provider(self) -> None:
        config = {"plugins": {"enabled": True}}

        result = configure(
            config,
            provider="sglang",
            model="Qwen/test",
            base_url="http://127.0.0.1:30100/v1",
            context_window=32768,
            max_tokens=1024,
        )

        self.assertTrue(result["plugins"]["enabled"])
        self.assertEqual(result["agents"]["defaults"]["model"]["primary"], "sglang/Qwen/test")
        provider = result["models"]["providers"]["sglang"]
        self.assertEqual(provider["baseUrl"], "http://127.0.0.1:30100/v1")
        self.assertEqual(provider["models"][0]["contextWindow"], 32768)


if __name__ == "__main__":
    unittest.main()
