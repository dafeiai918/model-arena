# -*- coding: utf-8 -*-
"""model-arena 测试：临时库 + 本地假站点，不发外网请求"""
import json
import sys
import tempfile
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import model_arena as ma  # noqa: E402


class _Api(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, *args):
        pass

    def do_GET(self):
        body = json.dumps({"data": [{"id": "mock-a"}, {"id": "mock-b"}]}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self):
        length = int(self.headers.get("Content-Length") or 0)
        payload = json.loads(self.rfile.read(length).decode("utf-8"))
        body = json.dumps({"choices": [{"message": {"content": f"来自 {payload['model']} 的回答"}}]}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


class ModelArenaTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.api = ThreadingHTTPServer(("127.0.0.1", 0), _Api)
        cls.api_port = cls.api.server_address[1]
        threading.Thread(target=cls.api.serve_forever, daemon=True).start()

    @classmethod
    def tearDownClass(cls):
        cls.api.shutdown()

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        base = Path(self.tmp.name)
        self._orig = (ma.DB_PATH, ma.CONFIG_PATH, ma.DATA_DIR)
        ma.DATA_DIR = base
        ma.DB_PATH = base / "arena.db"
        ma.CONFIG_PATH = base / "config.json"

    def tearDown(self):
        ma.DB_PATH, ma.CONFIG_PATH, ma.DATA_DIR = self._orig
        self.tmp.cleanup()

    def _setup(self):
        return ma.main(["setup", "-u", f"http://127.0.0.1:{self.api_port}/v1",
                        "-k", "sk-test", "-m", "mock-a", "mock-b"])

    # ---------- Elo ----------

    def test_expected_score_symmetry(self):
        self.assertAlmostEqual(ma.expected_score(1000, 1000), 0.5)
        self.assertAlmostEqual(ma.expected_score(1000, 1400) + ma.expected_score(1400, 1000), 1.0)
        self.assertGreater(ma.expected_score(1400, 1000), 0.9)

    def test_apply_result_zero_sum(self):
        a, b = ma.apply_result(1000, 1000, 1.0)
        self.assertAlmostEqual(a, 1016.0)
        self.assertAlmostEqual(b, 984.0)
        self.assertAlmostEqual(a + b, 2000.0)

    def test_upset_gains_more(self):
        low_up, high_down = ma.apply_result(1000, 1400, 1.0)
        self.assertGreater(low_up - 1000, 20)

    # ---------- 存储 ----------

    def test_record_battle_updates_ratings(self):
        conn = ma.connect()
        ma.record_battle(conn, "问题", "a", "b", "a")
        row_a = conn.execute("SELECT rating, wins, losses FROM ratings WHERE model='a'").fetchone()
        row_b = conn.execute("SELECT rating, wins, losses FROM ratings WHERE model='b'").fetchone()
        self.assertAlmostEqual(row_a[0], 1016.0)
        self.assertEqual(row_a[1], 1)
        self.assertEqual(row_b[2], 1)
        conn.close()

    def test_record_draw(self):
        conn = ma.connect()
        ma.record_battle(conn, "问题", "a", "b", "draw")
        row = conn.execute("SELECT rating, draws FROM ratings WHERE model='a'").fetchone()
        self.assertAlmostEqual(row[0], 1000.0)
        self.assertEqual(row[1], 1)
        conn.close()

    def test_board_after_battles(self):
        conn = ma.connect()
        ma.record_battle(conn, "q1", "a", "b", "a")
        ma.record_battle(conn, "q2", "a", "b", "a")
        conn.close()
        self.assertEqual(ma.main(["board"]), 0)

    def test_board_empty(self):
        self.assertEqual(ma.main(["board"]), 1)

    # ---------- 配置 ----------

    def test_setup_and_models_management(self):
        self.assertEqual(self._setup(), 0)
        self.assertEqual(ma.load_config()["models"], ["mock-a", "mock-b"])
        self.assertEqual(ma.main(["models", "add", "mock-c"]), 0)
        self.assertIn("mock-c", ma.load_config()["models"])
        self.assertEqual(ma.main(["models", "remove", "mock-c"]), 0)
        self.assertNotIn("mock-c", ma.load_config()["models"])
        self.assertEqual(ma.main(["models", "clear"]), 0)
        self.assertEqual(ma.load_config()["models"], [])

    def test_models_add_requires_names(self):
        self.assertEqual(ma.main(["models", "add"]), 1)

    # ---------- 请求 ----------

    def test_ask_model_ok(self):
        self._setup()
        cfg = ma.load_config()
        text, err = ma.ask_model(cfg, "mock-a", "你好")
        self.assertEqual(err, "")
        self.assertIn("mock-a", text)

    def test_ask_model_failure(self):
        cfg = dict(ma.DEFAULT_CONFIG, base_url="http://127.0.0.1:1/v1", api_key="sk-x")
        text, err = ma.ask_model(cfg, "m", "hi")
        self.assertEqual(text, "")
        self.assertTrue(err)

    def test_ask_both(self):
        self._setup()
        cfg = ma.load_config()
        outcome = ma.ask_both(cfg, "mock-a", "mock-b", "同一个问题")
        self.assertIn("mock-a", outcome["a"][1])
        self.assertIn("mock-b", outcome["b"][1])

    def test_fetch_models(self):
        self._setup()
        self.assertEqual(ma.main(["fetch"]), 0)

    def test_battle_requires_two_models(self):
        ma.main(["setup", "-u", f"http://127.0.0.1:{self.api_port}/v1", "-k", "sk-test"])
        self.assertEqual(ma.main(["battle", "问题"]), 1)

    def test_battle_requires_config(self):
        self.assertEqual(ma.main(["battle", "问题"]), 1)

    def test_reset(self):
        conn = ma.connect()
        ma.record_battle(conn, "q", "a", "b", "a")
        conn.close()
        self.assertEqual(ma.main(["reset"]), 0)
        self.assertEqual(ma.main(["board"]), 1)


if __name__ == "__main__":
    unittest.main(verbosity=2)
