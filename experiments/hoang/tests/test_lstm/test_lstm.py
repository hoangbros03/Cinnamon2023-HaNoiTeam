"""
LSTM test file
"""
import logging
import unittest

import torch
import torch.nn as nn

import experiments.hoang.models.lstm_htb as lstm_htb
from experiments.hoang.tests.test_utils import (
    CONSTANT_VARIABLES,
    get_and_process_data,
    get_model,
    train,
)

# Logging config
logging.basicConfig(format="%(asctime)s %(message)s")
log = logging.getLogger("test_logger")
log.setLevel(logging.DEBUG)


# Test classes
class TestLSTMOneToOneModel(unittest.TestCase):
    """
    Class handle tests for LSTM one to one model
    """

    def test_type_of_model(self):
        """
        Test type of the model
        """
        model = get_model("LSTM", 100, 100, 100)
        self.assertIsInstance(model, lstm_htb.LSTM)

    def test_init_model(self):
        """
        Test model error when initialized
        """
        get_model("LSTM", 1, 32, 1, activation1="tanh")
        get_model("LSTM", 5, 128, 16)
        get_model("LSTM", 8, 64, 1, activation2="naunaunaunau")
        with self.assertRaises(TypeError):
            get_model("LSTM", 1, 512, 1, bias="ok")
        with self.assertRaises(ValueError):
            get_model("LSTM", -1, 512, 1)
        with self.assertRaises(TypeError):
            get_model("LSTM", True, 16, 2)
        with self.assertRaises(ValueError):
            get_model("LSTM", 1, 32, -6)
        with self.assertRaises(ValueError):
            get_model("LSTM", 0, 0, 2)
        with self.assertRaises(ValueError):
            get_model("LSTM", 1, 2, 0)
        with self.assertRaises(TypeError):
            get_model("LSTM", 1, "ok", "ok")

    def test_output_of_model(self):
        """
        Test output of the model
        """
        model = get_model("LSTM", 1, 32, 4)
        x_test = torch.rand(1, 1)
        y_test = model.forward(x_test)
        self.assertTrue(list(y_test.shape) == list(torch.rand(1, 4).shape))
        model = get_model("LSTM", 3, 32, 8)
        x_test = torch.rand(3, 3)
        y_test = model.forward(x_test)
        self.assertTrue(list(y_test.shape) == list(torch.rand(3, 8).shape))

    def test_converage_on_a_sample(self):
        """
        Test converage on a sample
        """
        model = get_model("LSTM", 3, 32, 8, bias=True)
        x_test = torch.rand(1, 3)
        y_test = torch.tensor([4])
        y_pred = model.forward(x_test)
        initial_loss = nn.CrossEntropyLoss()(y_pred, y_test)
        for _ in range(CONSTANT_VARIABLES["ONE_TO_ONE_A_SAMPLE_EPOCHS"]):
            model.reset_states()
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
        model = get_model("LSTM", 4, 32, 3)
        y_pred = model.forward(X_test)
        initial_loss = nn.CrossEntropyLoss()(y_pred, y_test) / len(y_test)
        log.debug(f"Initial loss: {initial_loss}")
        for _ in range(CONSTANT_VARIABLES["ONE_TO_ONE_A_DATASET_EPOCHS"]):
            model.reset_states()
            model = train(
                model,
                X_train,
                y_train,
                nn.CrossEntropyLoss(),
                torch.optim.Adam(params=model.parameters(), lr=0.001),
                False,
            )
        model.reset_states()
        y_pred = model.forward(X_test)
        final_loss = nn.CrossEntropyLoss()(y_pred, y_test) / len(y_test)
        log.debug(f"Final loss: {final_loss}")
        self.assertLess(final_loss, initial_loss)


class TestLSTMManyToOneModel(unittest.TestCase):
    """
    Class holds test for LSTM many to one model
    """

    def test_type_of_model(self):
        """
        Test type of the model
        """
        model = get_model("manyToOneLSTM", 100, 100, 100, 100)
        self.assertIsInstance(model, lstm_htb.manyToOneLSTM)

    def test_init_model(self):
        """
        Test model when initialized
        """
        get_model("manyToOneLSTM", 1, 32, 1, 5)
        get_model("manyToOneLSTM", 5, 128, 16, 1)
        get_model("manyToOneLSTM", 5, 128, 16, 1, activation1="ReLU")
        with self.assertRaises(ValueError):
            get_model("manyToOneLSTM", -1, 512, 1, 0)
        with self.assertRaises(TypeError):
            get_model("manyToOneLSTM", 16, 215, 10, 2, bias="ratoke")
        with self.assertRaises(ValueError):
            get_model("manyToOneLSTM", -1, 512, 1, 0)
        with self.assertRaises(ValueError):
            get_model("manyToOneLSTM", 18, 512, 1, -8)
        with self.assertRaises(TypeError):
            get_model("manyToOneLSTM", True, 16, 2, 2)
        with self.assertRaises(ValueError):
            get_model("manyToOneLSTM", 1, 32, -6, 8)
        with self.assertRaises(ValueError):
            get_model("manyToOneLSTM", 1, 0, 0, 16)
        with self.assertRaises(ValueError):
            get_model("manyToOneLSTM", 0, 2, 8, 32)
        with self.assertRaises(TypeError):
            get_model("manyToOneLSTM", "ok", "ok", 17, 32)

    def test_output_of_model(self):
        """
        Test output of the model
        """
        model = get_model("manyToOneLSTM", 1, 32, 4, 3)
        x_test = torch.rand(1, 3, 1)
        y_test = model.forward(x_test)
        log.debug(f"y_test shape: {y_test.shape}")
        self.assertTrue(list(y_test.shape) == list(torch.rand(1, 4).shape))
        model = get_model("manyToOneLSTM", 3, 32, 8, 9)
        x_test = torch.rand(3, 9, 3)
        y_test = model.forward(x_test)
        log.debug(f"y_test shape: {y_test.shape}")
        self.assertTrue(list(y_test.shape) == list(torch.rand(3, 8).shape))

    def test_converage_on_a_sample(self):
        """
        Test converage on a sample
        """
        model = get_model("manyToOneLSTM", 1, 32, 16, 3)
        x_test = torch.rand(1, 3, 1)
        y_test = torch.tensor([1])
        y_pred = model.forward(x_test)
        initial_loss = nn.CrossEntropyLoss()(y_pred, y_test)
        optimizer = torch.optim.Adam(params=model.parameters(), lr=0.001)
        for _ in range(CONSTANT_VARIABLES["MANY_TO_ONE_A_SAMPLE_EPOCHS"]):
            model.reset_states()
            model = train(
                model,
                x_test,
                y_test,
                nn.CrossEntropyLoss(),
                optimizer,
                False,
            )
        y_pred = model.forward(x_test)
        final_loss = nn.CrossEntropyLoss()(y_pred, y_test)
        # log.info(f"Initial loss: {initial_loss}, final_loss = {final_loss}")
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
        model = get_model("manyToOneLSTM", 2, 32, 3, 2)
        y_pred = model.forward(X_test)
        initial_loss = nn.CrossEntropyLoss()(y_pred, y_test) / len(y_test)
        log.debug(f"Initial loss: {initial_loss}")
        for _ in range(CONSTANT_VARIABLES["MANY_TO_ONE_A_DATASET_EPOCHS"]):
            model.reset_states()
            model = train(
                model,
                X_train,
                y_train,
                nn.CrossEntropyLoss(),
                torch.optim.Adam(params=model.parameters(), lr=0.001),
                False,
            )
        model.reset_states()
        y_pred = model.forward(X_test)
        final_loss = nn.CrossEntropyLoss()(y_pred, y_test) / len(y_test)
        log.debug(f"Final loss: {final_loss}")
        self.assertLess(final_loss, initial_loss)


class TestLSTMOneToManyModel(unittest.TestCase):
    """
    Class to test LSTM one to many model
    """

    def test_type_of_model(self):
        """
        Test type of the model
        """
        model = get_model("oneToManyLSTM", 1, 16, 1, output_times=2)
        self.assertIsInstance(model, lstm_htb.oneToManyLSTM)

    def test_init_model(self):
        """
        Test when model is initialized
        """
        get_model("oneToManyLSTM", 1, 32, 1, output_times=3, activation1="tanh")
        get_model("oneToManyLSTM", 5, 128, 5, output_times=1)
        get_model("oneToManyLSTM", 8, 64, 8, output_times=9, activation2="naunaunaunau")
        with self.assertRaises(TypeError):
            get_model("oneToManyLSTM", 5, 16, 5, output_times=0, bool="true")
        with self.assertRaises(TypeError):
            get_model("oneToManyLSTM", "9", 6, 9, output_times=1)
        with self.assertRaises(TypeError):
            get_model("oneToManyLSTM", 9, 6, 9, output_times="1")
        with self.assertRaises(TypeError):
            get_model("oneToManyLSTM", True, True, 9, output_times=2)
        with self.assertRaises(TypeError):
            get_model("oneToManyLSTM", 9, 2, 9, output_times=True)
        with self.assertRaises(ValueError):
            get_model("oneToManyLSTM", 6, 6, 6, output_times=0)
        with self.assertRaises(ValueError):
            get_model("oneToManyLSTM", 6, 12, 7, output_times=1)
        with self.assertRaises(ValueError):
            get_model("oneToManyLSTM", 7, 12, 7, output_times=-5)
        with self.assertRaises(ValueError):
            get_model("oneToManyLSTM", -7, -12, 7, output_times=5)
        with self.assertRaises(ValueError):
            get_model("oneToManyLSTM", -7, 12, -7, output_times=2)

    def test_output_of_model(self):
        """
        Test output of the model
        """
        x_test = torch.reshape(torch.tensor([4.0]), (1, 1, 1))
        y_test = torch.reshape(torch.tensor([0.1, 0.2, 0.3, 0.4]), (1, 4, 1))
        model = get_model("oneToManyLSTM", 1, 4, 1, output_times=4, bias=True)
        y_pred = model.forward(x_test)
        self.assertTrue(list(y_pred.shape) == list(y_test.shape))

    def test_converage_on_a_sample(self):
        """
        Test converage on a sample
        """
        x_test = torch.reshape(torch.tensor([0.5]), (1, 1, 1))
        y_test = torch.reshape(torch.tensor([0.1, 0.2, 0.3, 0.4]), (1, 4, 1))
        model = get_model("oneToManyLSTM", 1, 4, 1, output_times=4, bias=True)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.03)
        initial_loss = nn.MSELoss()(model.forward(x_test), y_test)
        log.info(f"Initial loss OTM sample: {initial_loss}")
        for _ in range(CONSTANT_VARIABLES["ONE_TO_MANY_A_SAMPLE_EPOCHS"]):
            train(
                model,
                x_test,
                y_test,
                nn.MSELoss(),
                optimizer,
                False,
            )
        final_loss = nn.MSELoss()(model.forward(x_test), y_test)
        log.info(f"Result: {model.forward(x_test)}")
        log.info(f"Final loss OTM sample: {final_loss}")
        self.assertLess(final_loss, initial_loss)

    def test_converage_on_a_dataset(self):
        """
        Test converage on a dataset
        """
        X_train, y_train, X_val, y_val, X_test, y_test = get_and_process_data(
            "oneToMany", 0.3, 0.1
        )
        log.info(f"X_train shape: {X_train.shape}")
        log.info(f"y_train shape: {y_train.shape}")
        model = get_model(
            "oneToManyLSTM",
            1,
            8,
            1,
            output_times=4,
            bias=True,
            activation1="tanh",
            activation2="tanh",
        )
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        initial_loss = nn.MSELoss()(model.forward(X_test), y_test)
        log.info(f"Initial loss OTM dataset: {initial_loss}")

        # Train on train dataset
        for _ in range(CONSTANT_VARIABLES["ONE_TO_MANY_A_DATASET_EPOCHS"]):
            train(
                model,
                X_test,
                y_test,
                nn.MSELoss(),
                optimizer,
                False,
            )

        # Eval
        final_loss = nn.MSELoss()(model.forward(X_test), y_test)
        log.info(f"Final loss OTM dataset: {final_loss}")
        self.assertLess(final_loss, initial_loss)


class TestLSTMManyToManyModel(unittest.TestCase):
    """
    Class to test LSTM many to many model
    """

    def test_type_of_model(self):
        """
        Test type of the model
        """
        model = get_model("manyToManyLSTM", 2, 16, 2, 3, 3, simultaneous=False)
        # log.info(dict(model.named_parameters()))
        self.assertIsInstance(model, lstm_htb.manyToManyLSTM)

    def test_init_model(self):
        """
        Test when model is initialized
        """
        # Cases shouldn't raise errors
        get_model(
            "manyToManyLSTM",
            2,
            16,
            2,
            output_times=3,
            input_times=4,
            activation1="ahuhu",
            simultaneous=False,
        )
        get_model("manyToManyLSTM", 2, 1, 3, 10, 10, bias=False, simultaneous=True)
        get_model(
            "manyToManyLSTM", 1, 16, 1, 1, 1, activation2="tanh", simultaneous=True
        )

        # Cases should raise error
        with self.assertRaises(TypeError):
            get_model(
                "manyToManyLSTM", 2, 1, 2.5, 10, 10, bias="False", simultaneous=True
            )
        with self.assertRaises(TypeError):
            get_model("manyToManyLSTM", 2, 1, 2, 10, 4, bias=True, simultaneous="2")
        with self.assertRaises(TypeError):
            get_model("manyToManyLSTM", 2, 3, 2, 10, "4", bias=True, simultaneous=True)
        with self.assertRaises(TypeError):
            get_model("manyToManyLSTM", 2, 3, 2, [10], 4, bias=True, simultaneous=True)
        with self.assertRaises(TypeError):
            get_model("manyToManyLSTM", {"number": 2}, 3, 2, 10, 4, simultaneous=True)
        with self.assertRaises(ValueError):
            get_model("manyToManyLSTM", 2, 1, 1, 3, 4)
        with self.assertRaises(ValueError):
            get_model("manyToManyLSTM", 0, 1, 0, 3, 4)
        with self.assertRaises(ValueError):
            get_model("manyToManyLSTM", 2, 1, 2, 3, 4, simultaneous=True)
        with self.assertRaises(ValueError):
            get_model("manyToManyLSTM", 2, 0, -2, 3, 4, simultaneous=False)
        with self.assertRaises(ValueError):
            get_model("manyToManyLSTM", 2, 8, 2, 3, 0, simultaneous=False)

    def test_output_of_model_no_simultaneous(self):
        """
        Test output of the model with no simuultaneous
        """
        X_test = torch.rand(1, 4, 3)
        y_shape = list(torch.rand(1, 8, 3).shape)
        model = get_model(
            "manyToManyLSTM", 3, 16, 3, 4, 8, activation1="tanh", simultaneous=False
        )
        model_output_shape = list(model.forward(X_test).shape)
        self.assertTrue(y_shape == model_output_shape)

    def test_output_of_model_with_simultaneous(self):
        """
        Test output of the model with simuultaneous
        """
        X_test = torch.rand(1, 4, 3)
        y_shape = list(torch.rand(1, 4, 3).shape)
        model = get_model(
            "manyToManyLSTM", 3, 16, 3, 4, 4, activation1="tanh", simultaneous=True
        )
        model_output_shape = list(model.forward(X_test).shape)
        self.assertTrue(y_shape == model_output_shape)

    def test_converage_on_a_sample_no_simultaneous(self):
        """
        Test converage on a sample with no simultaneous
        """
        X_test = torch.rand(1, 4, 3)
        y_test = torch.rand(1, 8, 3)
        model = get_model(
            "manyToManyLSTM",
            3,
            16,
            3,
            4,
            8,
            activation1="tanh",
            simultaneous=False,
            bias=True,
        )
        loss = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        initial_loss = loss(model.forward(X_test), y_test)
        for _ in range(CONSTANT_VARIABLES["MANY_TO_MANY_NO_SIMULTANEOUS_A_SAMPLE"]):
            train(model, X_test, y_test, loss, optimizer)
        final_loss = loss(model.forward(X_test), y_test)

        log.info(
            f"Initial loss MTMNS sample: {initial_loss}, "
            "final loss MTMNS sample: {final_loss}"
        )
        self.assertLess(final_loss, initial_loss)

    def test_converage_on_a_sample_with_simultaneous(self):
        """
        Test converage on a sample with simultaneous
        """
        X_test = torch.rand(1, 4, 3)
        y_test = torch.rand(1, 4, 3)
        model = get_model(
            "manyToManyLSTM", 3, 16, 3, 4, 4, activation1="tanh", simultaneous=True
        )
        loss = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        initial_loss = loss(model.forward(X_test), y_test)
        for _ in range(CONSTANT_VARIABLES["MANY_TO_MANY_SIMULTANEOUS_A_SAMPLE"]):
            train(model, X_test, y_test, loss, optimizer)
        final_loss = loss(model.forward(X_test), y_test)
        log.info(
            f"Initial loss MTMS sample: {initial_loss}, "
            "final loss MTMS sample: {final_loss}"
        )
        self.assertLess(final_loss, initial_loss)

    def test_converage_on_a_dataset_no_simultaneous(self):
        """
        Test converage on a dataset with no simultaneous
        """
        X_train, y_train, X_val, y_val, X_test, y_test = get_and_process_data(
            "manyToMany", 0.3, 0.1, False
        )
        log.info(f"X_train shape: {X_train.shape}")
        log.info(f"y_train shape: {y_train.shape}")
        model = get_model(
            "manyToManyLSTM",
            5,
            8,
            5,
            input_times=4,
            output_times=8,
            bias=True,
            activation1="tanh",
            activation2="tanh",
            simultaneous=False,
        )
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        initial_loss = nn.MSELoss()(model.forward(X_test), y_test)
        log.info(f"Initial loss MTMNS dataset: {initial_loss}")

        # Train on train dataset
        for _ in range(CONSTANT_VARIABLES["MANY_TO_MANY_NO_SIMULTANEOUS_A_DATASET"]):
            train(
                model,
                X_test,
                y_test,
                nn.MSELoss(),
                optimizer,
                False,
            )

        # Eval
        final_loss = nn.MSELoss()(model.forward(X_test), y_test)
        log.info(f"Final loss MTMNS dataset: {final_loss}")
        self.assertLess(final_loss, initial_loss)

    def test_converage_on_a_dataset_with_simultaneous(self):
        """
        Test converage on a dataset with simultaneous
        """
        X_train, y_train, X_val, y_val, X_test, y_test = get_and_process_data(
            "manyToMany", 0.3, 0.1, True
        )
        log.info(f"X_train shape: {X_train.shape}")
        log.info(f"y_train shape: {y_train.shape}")
        model = get_model(
            "manyToManyLSTM",
            5,
            8,
            5,
            input_times=4,
            output_times=4,
            bias=True,
            activation1="tanh",
            activation2="tanh",
            simultaneous=True,
        )
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        initial_loss = nn.MSELoss()(model.forward(X_test), y_test)
        log.info(f"Initial loss MTMS dataset: {initial_loss}")

        # Train on train dataset
        for _ in range(CONSTANT_VARIABLES["MANY_TO_MANY_SIMULTANEOUS_A_DATASET"]):
            train(
                model,
                X_test,
                y_test,
                nn.MSELoss(),
                optimizer,
                False,
            )

        # Eval
        final_loss = nn.MSELoss()(model.forward(X_test), y_test)
        log.info(f"Final loss MTMS dataset: {final_loss}")
        self.assertLess(final_loss, initial_loss)


if __name__ == "__main__":
    unittest.main()
