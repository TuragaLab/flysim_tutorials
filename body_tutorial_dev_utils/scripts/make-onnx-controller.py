#!/usr/bin/env python3
"""
Convert the trained walking controller hosted at
https://janelia.figshare.com/articles/dataset/25309105 to the ONNX format.
"""
from pathlib import Path
from shutil import unpack_archive
from tempfile import TemporaryDirectory
from typing import Any, Iterator, Sequence, cast
from urllib.request import urlretrieve

import acme.tf.utils
import onnx
import onnx.helper as oh
import onnx.numpy_helper as onh
import numpy as np
import onnx.reference
import tensorflow as tf
import tensorflow_probability.python.distributions as _
from flybody.fly_envs import walk_imitation


def main() -> None:
    download_policy_dir_if_missing()
    onnx_controller = get_onnx_controller()
    test_onnx_controller(onnx_controller)
    store_onnx_controller(onnx_controller)


def download_policy_dir_if_missing() -> None:
    url = "https://janelia.figshare.com/ndownloader/files/44815195"
    output_path = Path("_inbox/trained-fly-policies")

    if not output_path.exists():
        with TemporaryDirectory() as temp:
            urlretrieve(url, Path(temp, "archive.zip"))
            unpack_archive(Path(temp, "archive.zip"), output_path)


def get_onnx_controller() -> onnx.ModelProto:
    FLOAT = onnx.TensorProto.FLOAT

    params = controller_params()
    n_input_chan = params["block1/weights"].shape[0]

    input_info = [
        oh.make_tensor_value_info("input", FLOAT, [None, n_input_chan]),
    ]
    nodes = [
        *tanh_block_nodes("block1", input_name="input"),
        *elu_block_nodes("block2", input_name="block1/output"),
        *elu_block_nodes("block3", input_name="block2/output"),
        *elu_block_nodes("block4", input_name="block3/output"),
        *dist_gen_block_nodes("block5", input_name="block4/output"),
    ]
    output_info = [
        oh.make_tensor_value_info(node.output[0], FLOAT, [None, None]) for node in nodes
    ]

    return oh.make_model(
        graph=oh.make_graph(
            name="walking_controller",
            nodes=nodes,
            inputs=input_info,
            outputs=output_info,
            initializer=[onh.from_array(p, n) for n, p in params.items()],
        )
    )


def controller_params() -> dict[str, np.ndarray]:
    param_names_in_order = [
        "block1/biases",
        "block1/weights",
        "block1/offsets",
        "block1/scales",
        "block2/biases",
        "block2/weights",
        "block3/biases",
        "block3/weights",
        "block4/biases",
        "block4/weights",
        "block5/mean_biases",
        "block5/mean_weights",
        "block5/std_biases",
        "block5/std_weights",
    ]

    controller_vars = cast(Sequence[tf.Variable], get_tf_controller()._variables)
    return {n: p.numpy() for n, p in zip(param_names_in_order, controller_vars)}


def get_tf_controller() -> Any:
    return tf.saved_model.load("_inbox/trained-fly-policies/walking")


def tanh_block_nodes(block_name: str, input_name: str) -> Iterator[onnx.NodeProto]:
    bn = block_name
    yield oh.make_node(
        name=f"{bn}/matmul",
        op_type="MatMul",
        inputs=[input_name, f"{bn}/weights"],
        outputs=[f"{bn}/internalvalue1"],
    )
    yield oh.make_node(
        op_type="Add",
        inputs=[f"{bn}/internalvalue1", f"{bn}/biases"],
        outputs=[f"{bn}/internalvalue2"],
    )
    yield oh.make_node(
        op_type="LayerNormalization",
        inputs=[f"{bn}/internalvalue2", f"{bn}/scales", f"{bn}/offsets"],
        outputs=[f"{bn}/internalvalue3"],
        epsilon=1e-5,
    )
    yield oh.make_node(
        op_type="Tanh",
        inputs=[f"{bn}/internalvalue3"],
        outputs=[f"{bn}/output"],
    )


def elu_block_nodes(block_name: str, input_name: str) -> Iterator[onnx.NodeProto]:
    bn = block_name
    yield oh.make_node(
        name=f"{bn}/matmul",
        op_type="MatMul",
        inputs=[input_name, f"{bn}/weights"],
        outputs=[f"{bn}/internalvalue1"],
    )
    yield oh.make_node(
        op_type="Add",
        inputs=[f"{bn}/internalvalue1", f"{bn}/biases"],
        outputs=[f"{bn}/internalvalue2"],
    )
    yield oh.make_node(
        op_type="Elu",
        inputs=[f"{bn}/internalvalue2"],
        outputs=[f"{bn}/output"],
    )


def dist_gen_block_nodes(block_name: str, input_name: str) -> Iterator[onnx.NodeProto]:
    bn = block_name
    yield oh.make_node(
        op_type="Constant",
        inputs=[],
        outputs=[f"{bn}/scale_gain"],
        value=float(0.7 / np.log(1.0 + np.exp(0.0))),
    )
    yield oh.make_node(
        op_type="Constant",
        inputs=[],
        outputs=[f"{bn}/min_scale"],
        value=1e-6,
    )
    yield oh.make_node(
        name=f"{bn}/matmul",
        op_type="MatMul",
        inputs=[input_name, f"{bn}/mean_weights"],
        outputs=[f"{bn}/internalvalue1"],
    )
    yield oh.make_node(
        op_type="MatMul",
        inputs=[input_name, f"{bn}/std_weights"],
        outputs=[f"{bn}/internalvalue2"],
    )
    yield oh.make_node(
        op_type="Add",
        inputs=[f"{bn}/internalvalue2", f"{bn}/std_biases"],
        outputs=[f"{bn}/internalvalue3"],
    )
    yield oh.make_node(
        op_type="Softplus",
        inputs=[f"{bn}/internalvalue3"],
        outputs=[f"{bn}/internalvalue4"],
    )
    yield oh.make_node(
        op_type="Mul",
        inputs=[f"{bn}/internalvalue4", f"{bn}/scale_gain"],
        outputs=[f"{bn}/internalvalue5"],
    )
    yield oh.make_node(
        op_type="Add",
        inputs=[f"{bn}/internalvalue1", f"{bn}/mean_biases"],
        outputs=["control_signal_means"],
    )
    yield oh.make_node(
        op_type="Add",
        inputs=[f"{bn}/internalvalue5", f"{bn}/min_scale"],
        outputs=["control_signal_stds"],
    )


def test_onnx_controller(onnx_controller: onnx.ModelProto) -> None:
    env = walk_imitation()
    timestep = env.reset()

    tf_input = acme.tf.utils.add_batch_dim(
        {k: v.astype(np.float32) for k, v in timestep.observation.items()}
    )
    onnx_input = np.concatenate(
        [v.numpy().reshape(1, -1) for _, v in sorted(tf_input.items())],
        axis=1,
    )

    tf_controller = get_tf_controller()
    tf_control_dist = tf_controller(tf_input)
    tf_control_means = tf_control_dist.mean()[0].numpy()
    tf_control_stds = tf_control_dist.stddev()[0].numpy()

    evaluator = onnx.reference.ReferenceEvaluator(onnx_controller)
    output_names = ["control_signal_means", "control_signal_stds"]
    onnx_control_dist = evaluator.run(output_names, {"input": onnx_input})
    onnx_control_means = onnx_control_dist[0][0]  # type: ignore
    onnx_control_stds = onnx_control_dist[1][0]  # type: ignore

    assert np.corrcoef(tf_control_means, onnx_control_means)[0, 1] >= 0.999
    assert np.corrcoef(tf_control_stds, onnx_control_stds)[0, 1] >= 0.999
    assert np.allclose(onnx_control_means, tf_control_means, rtol=1e-4, atol=1e-7)
    assert np.allclose(onnx_control_stds, tf_control_stds, rtol=1e-4, atol=1e-7)


def store_onnx_controller(onnx_controller: onnx.ModelProto) -> None:
    output_path = Path("_outbox/walking-controller.onnx")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_bytes(onnx_controller.SerializeToString())


if __name__ == "__main__":
    main()
