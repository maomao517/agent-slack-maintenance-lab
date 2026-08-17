import tempfile
import unittest
from pathlib import Path

from slackmaint.document_service import (
    DocumentAnalysisService,
    JsonlEventWriter,
    MockDocumentBackend,
    TransformersQwen3VLBackend,
)


def request(workflow_id: str = "w1", version: str = "v1"):
    return {
        "run_id": "run-1",
        "workflow_id": workflow_id,
        "task_id": workflow_id,
        "turn_id": 0,
        "document_id": "doc-1",
        "page_id": "page-1",
        "document_version": version,
        "question": "What is on this page?",
        "max_new_tokens": 8,
    }


class DocumentServiceTest(unittest.TestCase):
    def test_feature_injection_owner_prefers_inner_forward_model(self) -> None:
        class InnerModel:
            def get_image_features(self):
                return None

        class OuterModel:
            def __init__(self):
                self.model = InnerModel()

            def get_image_features(self):
                return self.model.get_image_features()

        backend = TransformersQwen3VLBackend.__new__(TransformersQwen3VLBackend)
        backend.model = OuterModel()
        backend.feature_owner = backend.model

        self.assertIs(
            backend._find_feature_injection_owner(),
            backend.model.model,
        )

    def test_shared_cache_reuses_state_across_workflows(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            image = Path(temporary) / "page.png"
            image.write_bytes(b"fake-image")
            backend = MockDocumentBackend()
            service = DocumentAnalysisService(
                backend,
                cache_policy="shared_cpu",
                cache_capacity_mb=1,
                event_writer=JsonlEventWriter(None),
            )

            first = service.analyze(request("w1"), image)
            second = service.analyze(request("w2"), image)

            self.assertFalse(first["cache_hit"])
            self.assertTrue(second["cache_hit"])
            self.assertEqual(backend.encoder_calls, 1)
            self.assertEqual(service.metrics()["cache_hits"], 1)

    def test_task_local_cache_does_not_cross_workflows(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            image = Path(temporary) / "page.png"
            image.write_bytes(b"fake-image")
            backend = MockDocumentBackend()
            service = DocumentAnalysisService(
                backend,
                cache_policy="task_local",
                cache_capacity_mb=1,
                event_writer=JsonlEventWriter(None),
            )

            service.analyze(request("w1"), image)
            service.analyze(request("w2"), image)

            self.assertEqual(backend.encoder_calls, 2)

    def test_document_version_invalidates_shared_state(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            image = Path(temporary) / "page.png"
            image.write_bytes(b"fake-image")
            backend = MockDocumentBackend()
            service = DocumentAnalysisService(
                backend,
                cache_policy="shared_cpu",
                cache_capacity_mb=1,
                event_writer=JsonlEventWriter(None),
            )

            service.analyze(request("w1", "v1"), image)
            second = service.analyze(request("w2", "v2"), image)

            self.assertFalse(second["cache_hit"])
            self.assertEqual(backend.encoder_calls, 2)

    def test_no_cache_always_encodes(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            image = Path(temporary) / "page.png"
            image.write_bytes(b"fake-image")
            backend = MockDocumentBackend()
            service = DocumentAnalysisService(
                backend,
                cache_policy="no_cache",
                cache_capacity_mb=1,
                event_writer=JsonlEventWriter(None),
            )

            service.analyze(request(), image)
            service.analyze(request(), image)

            self.assertEqual(backend.encoder_calls, 2)
            self.assertEqual(service.metrics()["cache_hits"], 0)


if __name__ == "__main__":
    unittest.main()
