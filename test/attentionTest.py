import logging
import unittest

import torch
import torch.nn as nn
from testUtil import A_DATASET_EPOCH, device, train_attention

from models.transformers.sub_layers import (  # isort: skip
    MultiheadAttention,
    SelfAttention,
)

# Logging config
logging.basicConfig(format="%(asctime)s %(message)s")
log = logging.getLogger("test_logger")
log.setLevel(logging.DEBUG)


class TestSelfAttention(unittest.TestCase):
    """
    Class to test self attention
    """

    def test_init_model_no_error(self):
        """
        Test model init without error
        """
        _ = SelfAttention(10, 3, 3, 4)
        _ = SelfAttention(20, 5, 5, 7)
        _ = SelfAttention(33, 1, 1, 1)

    def test_init_model_error(self):
        """
        Test model init with error
        """
        with self.assertRaises(ValueError):
            _ = SelfAttention(0, 0, 0, 0)
        with self.assertRaises(ValueError):
            _ = SelfAttention(10, -1, -1, -3)
        with self.assertRaises(TypeError):
            _ = SelfAttention("ok", "rat ok", "rat rat ok", "rat x3 ok")

    def test_output_shape1(self):
        """
        Test model output shape
        """
        model = SelfAttention(10, 3, 3, 4)
        output_shape = model.forward(
            torch.rand(10, 6, 10), torch.rand(10, 6, 10), torch.rand(10, 6, 10)
        ).shape
        good_output_shape = torch.rand(10, 6, 4).shape
        self.assertTrue(list(output_shape) == list(good_output_shape))

    def test_output_shape2(self):
        """
        Test model output shape
        """
        model = SelfAttention(1, 1, 1, 1)
        output_shape = model.forward(
            torch.rand(10, 6, 1), torch.rand(10, 6, 1), torch.rand(10, 6, 1)
        ).shape
        good_output_shape = torch.rand(10, 6, 1).shape
        self.assertTrue(list(output_shape) == list(good_output_shape))

    def test_converge(self):
        """
        Test converage of the model
        """
        model = SelfAttention(10, 3, 3, 4)
        x_test = torch.rand(10, 6, 10)
        y_test = torch.rand(10, 6, 4)

        # First forward
        model.train()
        model.to(device)
        loss_fn = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        y_pred = model.forward(x_test, x_test, x_test)
        initial_loss = loss_fn(y_pred, y_test)

        # Train process
        _, final_loss = train_attention(
            model, x_test, x_test, x_test, y_test, A_DATASET_EPOCH, optimizer, loss_fn
        )
        self.assertLess(final_loss, initial_loss)


class TestMultiheadAttention(unittest.TestCase):
    """
    Class to test multihead attention
    """

    def test_init_model_no_error(self):
        """
        Test model init no error
        """
        _ = MultiheadAttention(10, 5)
        _ = MultiheadAttention(1, 1)
        _ = MultiheadAttention(3, 3)

    def test_init_model_error(self):
        """
        Test model init with error
        """
        with self.assertRaises(ValueError):
            _ = MultiheadAttention(10, 3)
        with self.assertRaises(ValueError):
            _ = MultiheadAttention(5, 0)
        with self.assertRaises(TypeError):
            _ = MultiheadAttention(1, "3")

    def test_output_shape_1(self):
        """
        Test model output shape
        """
        model = MultiheadAttention(10, 5)
        output_shape = model.forward(
            torch.rand(10, 6, 10), torch.rand(10, 6, 10), torch.rand(10, 6, 10)
        ).shape
        good_output_shape = torch.rand(10, 6, 10).shape
        self.assertTrue(list(output_shape) == list(good_output_shape))

    def test_output_shape_2(self):
        """
        Test model output shape
        """
        model = MultiheadAttention(16, 2)
        output_shape = model.forward(
            torch.rand(10, 6, 16), torch.rand(10, 6, 16), torch.rand(10, 6, 16)
        ).shape
        good_output_shape = torch.rand(10, 6, 16).shape
        self.assertTrue(list(output_shape) == list(good_output_shape))

    def test_converge(self):
        """
        Test converge of the model
        """
        model = MultiheadAttention(10, 5)
        x_test = torch.rand(10, 6, 10)
        y_test = torch.rand(10, 6, 10)

        # First forward
        model.train()
        model.to(device)
        loss_fn = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        y_pred = model.forward(x_test, x_test, x_test)
        initial_loss = loss_fn(y_pred, y_test)

        # Train process
        _, final_loss = train_attention(
            model, x_test, x_test, x_test, y_test, A_DATASET_EPOCH, optimizer, loss_fn
        )
        self.assertLess(final_loss, initial_loss)


if __name__ == "__main__":
    unittest.main()
