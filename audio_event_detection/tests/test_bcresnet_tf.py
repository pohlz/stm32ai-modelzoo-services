"""BC-ResNet Keras save/reload and upstream block parity checks."""
import ast
import importlib.util
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest

import numpy as np
import tensorflow as tf

ROOT = Path(__file__).resolve().parents[1]
MODELS = ROOT / "tf/src/models"
spec = importlib.util.spec_from_file_location("kws_adapter", MODELS / "kws_streaming_model.py")
adapter = importlib.util.module_from_spec(spec)
spec.loader.exec_module(adapter)


def upstream_classes():
    """Execute pristine upstream class definitions without unused streaming/QAT imports."""
    namespace = {"tf": tf}
    for filename, names in (("sub_spectral_normalization.py", {"SubSpectralNormalization"}),
                            ("bc_resnet.py", {"TransitionBlock", "NormalBlock"})):
        tree = ast.parse((MODELS / "kws_streaming/upstream" / filename).read_text())
        classes = ast.Module(body=[n for n in tree.body if isinstance(n, ast.ClassDef) and n.name in names], type_ignores=[])
        exec(compile(classes, filename, "exec"), namespace)
        if "SubSpectralNormalization" in namespace:
            namespace["sub_spectral_normalization"] = SimpleNamespace(
                SubSpectralNormalization=namespace["SubSpectralNormalization"])
    return namespace


class KwsStreamingTests(unittest.TestCase):
    def test_evaluators_save_reports_when_gui_display_is_disabled(self):
        # Guard against the old regression where display_figures=false also
        # suppressed creation of confusion-matrix artifacts.
        for relative in ("tf/src/evaluation/keras_evaluator.py",
                         "tf/src/evaluation/tflite_evaluator.py"):
            source = (ROOT / relative).read_text()
            tree = ast.parse(source)
            evaluate = next(node for node in ast.walk(tree)
                            if isinstance(node, ast.FunctionDef) and node.name == "evaluate")
            calls = [node for node in ast.walk(evaluate) if isinstance(node, ast.Call)]
            self.assertTrue(any(isinstance(call.func, ast.Attribute)
                                and call.func.attr == "_display_figures" for call in calls))
            self.assertFalse(any(isinstance(node, ast.If)
                                 and "display_figures" in ast.unparse(node.test)
                                 for node in ast.walk(evaluate)))

    def test_upstream_block_parity(self):
        classes = upstream_classes()
        for transition in (True, False):
            inputs = tf.keras.Input((12, 20, 8))
            model = tf.keras.Model(inputs, adapter._block(
                inputs, 12 if transition else 8, (2, 1), (1, 2) if transition else (1, 1),
                0., "check", transition))
            original = classes["TransitionBlock" if transition else "NormalBlock"](
                filters=12 if transition else 8, dilation=(2, 1),
                stride=(1, 2) if transition else 1, dropout=0.)
            x = tf.random.normal((2, 12, 20, 8))
            original(x, training=False)
            mapping = {"frequency_dw": original.frequency_dw_conv,
                       "ssn_bn": original.spectral_norm.batch_norm,
                       "temporal_dw": original.temporal_dw_conv,
                       "temporal_bn": original.batch_norm2 if transition else original.batch_norm,
                       "pointwise": original.conv1x1_2 if transition else original.conv1x1}
            if transition:
                mapping.update(transition=original.conv1x1_1, transition_bn=original.batch_norm1)
            for name, layer in mapping.items():
                layer.set_weights(model.get_layer("check_" + name).get_weights())
            np.testing.assert_allclose(model(x, training=False), original(x, training=False), atol=2e-6, rtol=2e-5)

    def test_training_and_serialization(self):
        model = adapter.get_custom_model(12, (40, 96, 1), dropout=0.1)
        self.assertEqual(model.output_shape, (None, 12))
        self.assertEqual(sum(layer.name.endswith("_residual_add") for layer in model.layers), 16)
        model.compile(optimizer=tf.keras.optimizers.SGD(0.01), loss="categorical_crossentropy")
        x = tf.random.normal((2, 40, 96, 1))
        loss = model.train_on_batch(x, tf.one_hot([0, 11], 12))
        self.assertTrue(np.isfinite(loss))
        expected = model(x, training=False).numpy()
        np.testing.assert_allclose(expected.sum(axis=1), 1., atol=1e-6)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "model.keras"
            model.save(path)
            restored = tf.keras.models.load_model(path)
            np.testing.assert_allclose(expected, restored(x, training=False), atol=1e-6)


if __name__ == "__main__":
    unittest.main()
