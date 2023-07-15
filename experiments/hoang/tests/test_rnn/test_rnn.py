"""
RNN test file
"""
import logging
import unittest

import torch
import torch.nn as nn

import experiments.hoang.models.rnn_htb as rnn_htb
from experiments.hoang.tests.test_utils import get_and_process_data, get_model, train

# Logging config
logging.basicConfig(format="%(asctime)s %(message)s")
log = logging.getLogger("test_logger")
log.setLevel(logging.DEBUG)


# Test classes
class TestRNNOneToOneModel(unittest.TestCase):
    """
    Class handle tests for RNN one to one model
    """

    def test_type_of_model(self):
        """
        Test type of the model
        """
        model = get_model("RNN", 100, 100, 100)
        self.assertIsInstance(model, rnn_htb.RNN)

    def test_init_model(self):
        """
        Test model error when initialized
        """
        get_model("RNN", 1, 32, 1, activation1="tanh")
        get_model("RNN", 5, 128, 16)
        get_model("RNN", 8, 64, 1, activation2="naunaunaunau")
        with self.assertRaises(TypeError):
            get_model("RNN", 1, 512, 1, bias="ok")
        with self.assertRaises(ValueError):
            get_model("RNN", -1, 512, 1)
        with self.assertRaises(ValueError):
            get_model("RNN", 1, -99, 16)
        with self.assertRaises(TypeError):
            get_model("RNN", True, 16, 2)
        with self.assertRaises(ValueError):
            get_model("RNN", 1, 32, -6)
        with self.assertRaises(ValueError):
            get_model("RNN", 1, 0, 6)
        with self.assertRaises(ValueError):
            get_model("RNN", 1, 2, 0)
        with self.assertRaises(TypeError):
            get_model("RNN", 1, "ok", 17)

    def test_output_of_model(self):
        """
        Test output of the model
        """
        model = get_model("RNN", 1, 32, 4)
        x_test = torch.rand(1, 1)
        y_test = model.forward(x_test)
        self.assertTrue(list(y_test.shape) == list(torch.rand(1, 4).shape))
        model = get_model("RNN", 3, 32, 8)
        x_test = torch.rand(3, 3)
        y_test = model.forward(x_test)
        self.assertTrue(list(y_test.shape) == list(torch.rand(3, 8).shape))

    def test_converage_on_a_sample(self):
        """
        Test converage on a sample
        """
        model = get_model("RNN", 3, 32, 8)
        x_test = torch.rand(1, 3)
        y_test = torch.tensor([4])
        y_pred = model.forward(x_test)
        initial_loss = nn.CrossEntropyLoss()(y_pred, y_test)
        for _ in range(50):
            model.reset_a()
            model = train(
                model,
                x_test,
                y_test,
                nn.CrossEntropyLoss(),
                torch.optim.Adam(params=model.parameters(), lr=0.001),
                False,
            )
        y_pred = model.forward(x_test)
        final_loss = nn.CrossEntropyLoss()(y_pred, y_test)
        self.assertLess(final_loss, initial_loss)

    def test_converage_on_a_dataset(self):
        """
        Test converage on a dataset
        """
        X_train, y_train, X_val, y_val, X_test, y_test = get_and_process_data("iris")
        model = get_model("RNN", 4, 32, 3)
        y_pred = model.forward(X_test)
        initial_loss = nn.CrossEntropyLoss()(y_pred, y_test) / len(y_test)
        log.debug(f"Initial loss: {initial_loss}")
        for _ in range(500):
            model.reset_a()
            model = train(
                model,
                X_train,
                y_train,
                nn.CrossEntropyLoss(),
                torch.optim.Adam(params=model.parameters(), lr=0.001),
                False,
            )
        model.reset_a()
        y_pred = model.forward(X_test)
        final_loss = nn.CrossEntropyLoss()(y_pred, y_test) / len(y_test)
        log.debug(f"Final loss: {final_loss}")
        self.assertLess(final_loss, initial_loss)


class TestRNNManyToOneModel(unittest.TestCase):
    """
    Class holds test for RNN many to one model
    """

    def test_type_of_model(self):
        """
        Test type of the model
        """
        model = get_model("manyToOneRNN", 100, 100, 100, 100)
        self.assertIsInstance(model, rnn_htb.manyToOneRNN)

    def test_init_model(self):
        """
        Test model when initialized
        """
        get_model("manyToOneRNN", 1, 32, 1, 5)
        get_model("manyToOneRNN", 5, 128, 16, 1)
        get_model("manyToOneRNN", 5, 128, 16, 1, activation1="ReLU")
        with self.assertRaises(ValueError):
            get_model("manyToOneRNN", -1, 512, 1, 0)
        with self.assertRaises(TypeError):
            get_model("manyToOneRNN", 16, 215, 10, 2, bias="ratoke")
        with self.assertRaises(ValueError):
            get_model("manyToOneRNN", -1, 512, 1, 0)
        with self.assertRaises(ValueError):
            get_model("manyToOneRNN", 18, 512, 1, -8)
        with self.assertRaises(ValueError):
            get_model("manyToOneRNN", 1, -99, 16, 5)
        with self.assertRaises(TypeError):
            get_model("manyToOneRNN", True, 16, 2, 2)
        with self.assertRaises(ValueError):
            get_model("manyToOneRNN", 1, 32, -6, 8)
        with self.assertRaises(ValueError):
            get_model("manyToOneRNN", 1, 0, 6, 16)
        with self.assertRaises(ValueError):
            get_model("manyToOneRNN", 1, 2, 0, 32)
        with self.assertRaises(TypeError):
            get_model("manyToOneRNN", 1, "ok", 17, 32)

    def test_output_of_model(self):
        """
        Test output of the model
        """
        model = get_model("manyToOneRNN", 1, 32, 4, 3)
        x_test = torch.rand(1, 3, 1)
        y_test = model.forward(x_test)
        log.debug(f"y_test shape: {y_test.shape}")
        self.assertTrue(list(y_test.shape) == list(torch.rand(1, 4).shape))
        model = get_model("manyToOneRNN", 3, 32, 8, 9)
        x_test = torch.rand(3, 9, 3)
        y_test = model.forward(x_test)
        log.debug(f"y_test shape: {y_test.shape}")
        self.assertTrue(list(y_test.shape) == list(torch.rand(3, 8).shape))

    def test_converage_on_a_sample(self):
        """
        Test converage on a sample
        """
        model = get_model("manyToOneRNN", 1, 32, 16, 3)
        x_test = torch.rand(1, 3, 1)
        y_test = torch.tensor([1])
        y_pred = model.forward(x_test)
        initial_loss = nn.CrossEntropyLoss()(y_pred, y_test)
        for _ in range(50):
            model.reset_a()
            model = train(
                model,
                x_test,
                y_test,
                nn.CrossEntropyLoss(),
                torch.optim.Adam(params=model.parameters(), lr=0.001),
                False,
            )
        y_pred = model.forward(x_test)
        final_loss = nn.CrossEntropyLoss()(y_pred, y_test)
        self.assertLess(final_loss, initial_loss)

    def test_converage_on_a_dataset(self):
        """
        Test converage on a dataset
        """
        X_train, y_train, X_val, y_val, X_test, y_test = get_and_process_data("iris")
        X_train, X_test = torch.reshape(X_train, (-1, 2, 2)), torch.reshape(
            X_test, (-1, 2, 2)
        )
        log.info(f"X_train shape: {X_train.shape}")
        log.info(f"X_test shape: {X_test.shape}")
        model = get_model("manyToOneRNN", 2, 32, 3, 2)
        y_pred = model.forward(X_test)
        initial_loss = nn.CrossEntropyLoss()(y_pred, y_test) / len(y_test)
        log.debug(f"Initial loss: {initial_loss}")
        for _ in range(500):
            model.reset_a()
            model = train(
                model,
                X_train,
                y_train,
                nn.CrossEntropyLoss(),
                torch.optim.Adam(params=model.parameters(), lr=0.001),
                False,
            )
        model.reset_a()
        y_pred = model.forward(X_test)
        final_loss = nn.CrossEntropyLoss()(y_pred, y_test) / len(y_test)
        log.debug(f"Final loss: {final_loss}")
        self.assertLess(final_loss, initial_loss)


"""
Incomplete test classes
"""


class TestRNNOneToManyModel(unittest.TestCase):
    """
    Class to test RNN one to many model
    """

    def test_type_of_model(self):
        """
        Test type of the model
        """
        pass

    def test_init_model(self):
        """
        Test when model is initialized
        """
        pass

    def test_output_of_model(self):
        """
        Test output of the model
        """
        pass

    def test_converage_on_a_sample(self):
        """
        Test converage on a sample
        """
        pass

    def test_converage_on_a_dataset(self):
        """
        Test converage on a dataset
        """
        pass


class TestRNNManyToManyModel(unittest.TestCase):
    """
    Class to test RNN many to many model
    """

    def test_type_of_model(self):
        """
        Test type of the model
        """
        pass

    def test_init_model(self):
        """
        Test when model is initialized
        """
        pass

    def test_output_of_model_no_simultaneous(self):
        """
        Test output of the model with no simuultaneous
        """
        pass

    def test_output_of_model_with_simultaneous(self):
        """
        Test output of the model with simuultaneous
        """
        pass

    def test_converage_on_a_sample_no_simultaneous(self):
        """
        Test converage on a sample with no simultaneous
        """
        pass

    def test_converage_on_a_sample_with_simultaneous(self):
        """
        Test converage on a sample with simultaneous
        """
        pass

    def test_converage_on_a_dataset_no_simultaneous(self):
        """
        Test converage on a dataset with no simultaneous
        """
        pass

    def test_converage_on_a_dataset_with_simultaneous(self):
        """
        Test converage on a dataset with simultaneous
        """
        pass


if __name__ == "__main__":
    unittest.main()
