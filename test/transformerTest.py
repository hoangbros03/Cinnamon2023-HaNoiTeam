import unittest

import torch
import torch.nn as nn
from models.transformers.layers import DecoderLayer, EncoderLayer

# from models.transformers.model import Transformer
from testUtil import A_DATASET_EPOCH, device, train


class TestEncoderLayer(unittest.TestCase):
    """
    Class to test encoder layer
    """

    def test_init_no_error(self):
        """
        Test model init without error
        """
        _ = EncoderLayer(10, 5, 16)
        _ = EncoderLayer(16, 5, 16)
        _ = EncoderLayer(1, 1, 1)

    def test_init_with_error(self):
        """
        Test model init with error
        """
        with self.assertRaises(ValueError):
            _ = EncoderLayer(10, 0, 0)
        with self.assertRaises(ValueError):
            _ = EncoderLayer(10, 3, -1)
        with self.assertRaises(ValueError):
            _ = EncoderLayer(1, 9, 9)
        with self.assertRaises(TypeError):
            _ = EncoderLayer("ok", 3, 3)
        with self.assertRaises(TypeError):
            _ = EncoderLayer(10, ["1"], 1)

    def test_output_shape_1(self):
        """
        Test output of the model
        """
        model = EncoderLayer(10, 5, 16)
        output_shape = model.forward(torch.rand(64, 20, 10)).shape
        good_output_shape = torch.rand(64, 20, 10).shape
        self.assertTrue(list(output_shape) == list(good_output_shape))

    def test_output_shape_2(self):
        """
        Test output of the model
        """
        model = EncoderLayer(1, 1, 1)
        output_shape = model.forward(torch.rand(64, 20, 1)).shape
        good_output_shape = torch.rand(64, 20, 1).shape
        self.assertTrue(list(output_shape) == list(good_output_shape))

    def test_converge(self):
        """
        Test converge of the model
        """
        model = EncoderLayer(10, 5, 16)
        x_test = torch.rand(10, 6, 10)
        y_test = torch.rand(10, 6, 10)

        # First forward
        model.train()
        model.to(device)
        loss_fn = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        y_pred = model.forward(x_test)
        initial_loss = loss_fn(y_pred, y_test)

        # Train process
        _, final_loss = train(
            model, x_test, y_test, A_DATASET_EPOCH, optimizer, loss_fn
        )
        self.assertLess(final_loss, initial_loss)


class TestDecoderLayer(unittest.TestCase):
    """
    Class to test decoder layer
    """

    def test_init_no_error(self):
        """
        Test init model no error
        """
        _ = DecoderLayer(16, 4, 3)
        _ = DecoderLayer(8, 8, 8)
        _ = DecoderLayer(1, 1, 1)

    def test_init_with_error(self):
        """
        Test init model with error
        """
        with self.assertRaises(ValueError):
            _ = DecoderLayer(16, 3, 3)
        with self.assertRaises(ValueError):
            _ = DecoderLayer(1, 0, 1)
        with self.assertRaises(TypeError):
            _ = DecoderLayer("1", "1", 1)
        with self.assertRaises(TypeError):
            _ = DecoderLayer(3, 3, [3])

    def test_output_shape_1(self):
        """
        Test output of the model
        """
        model = DecoderLayer(1, 1, 1)
        output_shape = model.forward(torch.rand(64, 20, 1), torch.rand(64, 10, 1)).shape
        good_output_shape = torch.rand(64, 20, 1).shape
        self.assertTrue(list(output_shape) == list(good_output_shape))

    def test_output_shape_2(self):
        """
        Test output of the model
        """
        model = DecoderLayer(16, 4, 3)
        output_shape = model.forward(
            torch.rand(16, 20, 16), torch.rand(16, 10, 16)
        ).shape
        good_output_shape = torch.rand(16, 20, 16).shape
        self.assertTrue(list(output_shape) == list(good_output_shape))

    def test_converge(self):
        """
        Test converge of the model
        """
        model = DecoderLayer(10, 5, 16)
        x_test1 = torch.rand(10, 6, 10)
        x_test2 = torch.rand(10, 3, 10)
        y_test = torch.rand(10, 6, 10)

        # First forward
        model.train()
        model.to(device)
        loss_fn = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        y_pred = model.forward(x_test1, x_test2)
        initial_loss = loss_fn(y_pred, y_test)

        # Train process
        for _ in range(A_DATASET_EPOCH):
            y_pred = model.forward(x_test1, x_test2)
            loss = loss_fn(y_pred, y_test)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        final_loss = loss_fn(model.forward(x_test1, x_test2), y_test)
        self.assertLess(final_loss, initial_loss)


# class TestTransformer(unittest.TestCase):
#     """
#     Class to test transformer
#     """

#     def test_init_no_error(self):
#         pass

#     def test_init_with_error(self):
#         pass

#     def test_output_shape_1(self):
#         pass

#     def test_output_shape_2(self):
#         pass

#     def test_converge(self):
#         pass


if __name__ == "__main__":
    unittest.main()
