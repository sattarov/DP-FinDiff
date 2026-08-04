import pytest
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from unittest.mock import MagicMock, patch

from findiff.model import FinDiff


@pytest.fixture
def mock_dt():
    mock_data_transformer = MagicMock()
    mock_data_transformer.categorical_cols = ['col1', 'col2']
    mock_data_transformer.numerical_cols = ['num1']
    mock_data_transformer.categorical_mapping_ = {'col1': {1, 2}, 'col2': {1, 2, 3}}
    mock_data_transformer.label_cardinality_ = None
    mock_data_transformer.embedding_mapping_ = {
        'col1': np.array([0, 1]),
        'col2': np.array([2, 3, 4])
    }
    return mock_data_transformer


@patch('findiff.model.FinDiffSynthesizer')
@patch('findiff.model.BaseDiffuser')
@patch('findiff.model.nn.MSELoss')
@patch('findiff.model.optim.Adam')
@patch('findiff.model.CosineAnnealingLR')
def test_decode_cat_emb_logits(mock_cos, mock_adam, mock_mse, mock_diffuser, mock_synth, mock_dt):
    # 2. Instantiate FinDiff
    model = FinDiff(data_transformer=mock_dt)
    
    # 3. Create input test data
    # Two categorical columns with batch size of 2
    cat_logits = [
        torch.tensor([[10.0, 1.0], [1.0, 10.0]]),            # Argmax: 0, 1
        torch.tensor([[1.0, 10.0, 1.0], [1.0, 1.0, 10.0]])   # Argmax: 1, 2
    ]
    
    # 4. Run the method and assert
    result = model.decode_cat_emb_logits(cat_logits)
    expected = np.array([[0, 3], [1, 4]])
    np.testing.assert_array_equal(result, expected)


@patch('findiff.model.FinDiffSynthesizer')
@patch('findiff.model.BaseDiffuser')
@patch('findiff.model.optim.Adam')
@patch('findiff.model.CosineAnnealingLR')
def test_findiff_init(mock_cos, mock_adam, mock_diffuser, mock_synth, mock_dt):
    model = FinDiff(data_transformer=mock_dt, cat_emb_dim=3)
    
    assert model.cat_cols_dim == 2
    assert model.cat_emb_dim == 3
    mock_synth.assert_called_once()
    mock_diffuser.assert_called_once()


@patch('findiff.model.FinDiffSynthesizer')
@patch('findiff.model.BaseDiffuser')
@patch('findiff.model.optim.Adam')
@patch('findiff.model.CosineAnnealingLR')
@patch('findiff.model.PrivacyEngine')
def test_findiff_init_with_dp(mock_pe, mock_cos, mock_adam, mock_diffuser, mock_synth, mock_dt):
    """Test that PrivacyEngine is initialized correctly when dp_params are provided."""
    # 1. Setup DP parameters and mock instances
    dp_params = {
        "target_epsilon": 5.0,
        "target_delta": 1e-6,
        "max_grad_norm": 0.5,
    }
    num_epochs = 50
    train_loader = MagicMock()

    # Configure the mocks for FinDiffSynthesizer and Adam to return specific mock instances.
    # These will be the objects passed to make_private_with_epsilon.
    mock_synthesizer_instance = MagicMock(spec=nn.Module)
    # The constructor calls .to(device), so we mock it to return the instance itself.
    mock_synthesizer_instance.to.return_value = mock_synthesizer_instance
    mock_optimizer_instance = MagicMock(spec=optim.Optimizer)

    mock_synth.return_value = mock_synthesizer_instance
    mock_adam.return_value = mock_optimizer_instance

    # Mock the return value of the PrivacyEngine instance
    mock_privacy_engine_instance = mock_pe.return_value
    # make_private_with_epsilon returns (module, optimizer, data_loader)
    # We need to ensure that the *returned* module and optimizer are different mocks
    # from the ones passed in, to simulate Opacus wrapping them.
    mock_wrapped_synthesizer = MagicMock(spec=nn.Module)
    mock_wrapped_optimizer = MagicMock(spec=optim.Optimizer)
    mock_privacy_engine_instance.make_private_with_epsilon.return_value = (
        mock_wrapped_synthesizer, mock_wrapped_optimizer, MagicMock()
    )

    # 2. Instantiate FinDiff with DP parameters
    model = FinDiff(
        data_transformer=mock_dt,
        dp_params=dp_params,
        train_dataloader=train_loader,
        num_epochs=num_epochs
    )

    # 3. Assert that PrivacyEngine was called with the correct arguments
    mock_pe.assert_called_once()
    mock_privacy_engine_instance.make_private_with_epsilon.assert_called_once_with(
        module=mock_synthesizer_instance, # This should be the instance returned by mock_synth
        optimizer=mock_optimizer_instance, # This should be the instance returned by mock_adam
        data_loader=train_loader,
        target_epsilon=dp_params["target_epsilon"],
        target_delta=dp_params["target_delta"],
        max_grad_norm=dp_params["max_grad_norm"],
        epochs=num_epochs,
    )
    
    # Additionally, verify that model.synthesizer and model.optimizer are now the wrapped versions
    assert model.synthesizer is mock_wrapped_synthesizer
    assert model.optimizer is mock_wrapped_optimizer


@patch('findiff.model.FinDiffSynthesizer')
@patch('findiff.model.BaseDiffuser')
@patch('findiff.model.optim.Adam')
@patch('findiff.model.CosineAnnealingLR')
def test_fit(mock_cos, mock_adam, mock_diffuser, mock_synth, mock_dt):
    model = FinDiff(data_transformer=mock_dt, num_epochs=2)
    model.train_epoch = MagicMock(return_value=0.25)
    
    dummy_dataloader = [1, 2]
    model.fit(dummy_dataloader)
    
    assert model.train_epoch.call_count == 2
    assert model.training_history["train_loss"] == [0.25, 0.25]


@patch('findiff.model.FinDiffSynthesizer')
@patch('findiff.model.BaseDiffuser')
@patch('findiff.model.optim.Adam')
@patch('findiff.model.CosineAnnealingLR')
def test_train_epoch(mock_cos, mock_adam, mock_diffuser, mock_synth, mock_dt):
    model = FinDiff(data_transformer=mock_dt, cat_decoding='distance', diffusion_t_sampling='adaptive')
    
    model.synthesizer = MagicMock()
    # Mock the nn.Embedding layer's forward call
    model.synthesizer.x_cat_emb.return_value = torch.zeros((2, 2, 2))
    model.synthesizer.return_value = torch.zeros((2, 5))
    
    model.diffuser = MagicMock()
    # The default is now 'adaptive'
    model.diffuser.sample_adaptive_timesteps.return_value = torch.tensor([1, 2])
    model.diffuser.add_gauss_noise.return_value = (torch.zeros((2, 5)), torch.zeros((2, 5)))
    
    mock_loss = MagicMock()
    mock_loss.detach.return_value.cpu.return_value.numpy.return_value = 0.5
    model.loss_fnc = MagicMock()
    model.loss_fnc.return_value.sum.return_value.mean.return_value = mock_loss
    
    optimizer = MagicMock()
    scheduler = MagicMock()
    scheduler.last_epoch = 1
    scheduler.T_max = 10
    
    batch = {
        "cat": torch.tensor([[0, 2], [1, 3]]),
        "num": torch.tensor([[0.1], [0.2]])
    }
    dataloader = [batch]
    
    loss = model.train_epoch(dataloader, optimizer, scheduler)
    
    optimizer.zero_grad.assert_called_once()
    optimizer.step.assert_called_once()
    scheduler.step.assert_called_once()
    assert loss == 0.5


@patch('findiff.model.FinDiffSynthesizer')
@patch('findiff.model.BaseDiffuser')
@patch('findiff.model.optim.Adam')
@patch('findiff.model.CosineAnnealingLR')
@patch.object(FinDiff, 'decode_sample')
def test_sample(mock_decode, mock_cos, mock_adam, mock_diffuser, mock_synth, mock_dt):
    model = FinDiff(data_transformer=mock_dt, batch_size_sample=2)
    model.synthesizer = MagicMock()
    # The dim_input is calculated in __init__ and used in sample, so we can't just mock it.
    # Instead, we let the real calculation happen and then check the call.
    # dim_input = len(numerical_cols) + cat_emb_dim * cat_cols_dim = 1 + 2 * 2 = 5
    model.synthesizer.return_value = torch.zeros((2, 5))
    
    model.diffuser = MagicMock()
    model.diffuser.total_steps = 3
    model.diffuser.p_sample_gauss.return_value = torch.zeros((2, 5))
    
    mock_decode.return_value = pd.DataFrame({'col1': [1, 2], 'num1': [0.1, 0.2]})
    
    df = model.sample(n_samples=2)
    
    assert len(df) == 2
    mock_decode.assert_called_once()
    assert model.diffuser.p_sample_gauss.call_count == 3


@patch('findiff.model.FinDiffSynthesizer')
@patch('findiff.model.BaseDiffuser')
@patch('findiff.model.optim.Adam')
@patch('findiff.model.CosineAnnealingLR')
def test_decode_sample(mock_cos, mock_adam, mock_diffuser, mock_synth, mock_dt):
    model = FinDiff(data_transformer=mock_dt, cat_emb_dim=2)
    model.decode_cat_emb_distance = MagicMock(return_value=np.array([[0, 2]]))
    
    mock_dt.inverse_transform.return_value = pd.DataFrame({'col1': [1], 'num1': [0.1]})
    
    sample = torch.tensor([[0.1, 0.2, 0.3, 0.4, 0.5]])
    res = model.decode_sample(sample)
    
    model.decode_cat_emb_distance.assert_called_once()
    mock_dt.inverse_transform.assert_called_once()
    assert isinstance(res, pd.DataFrame)
