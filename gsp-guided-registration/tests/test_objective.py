import unittest

import torch
import torch.nn.functional as F

from gsp_reg.modules.gsp_projector import GSPProjector
from gsp_reg.modules.soft_classified_loss import (
    SoftClassifiedLoss,
    _soft_gate_from_response,
    compute_stopo,
    compute_stopo_chunked,
)


class ObjectiveTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(7)

    def test_gate_endpoints(self):
        response = torch.tensor([0.0, 0.03, 0.115, 0.20, 0.30])
        expected = torch.tensor([0.0, 0.0, 0.5, 1.0, 1.0])
        self.assertTrue(torch.allclose(_soft_gate_from_response(response), expected))

    def test_mass_normalization_preserves_constant_signal(self):
        projector = GSPProjector(num_superpoints=8)
        xyz = torch.rand(1, 40, 3)
        features = torch.ones(1, 40, 16, requires_grad=True)
        delta, assignment = projector(xyz, features)
        self.assertTrue(torch.allclose(assignment.sum(1), torch.ones(1, 40)))
        self.assertLess(float(delta.max()), 1e-5)
        delta.sum().backward()
        self.assertTrue(torch.isfinite(features.grad).all())

    def test_geometry_gate_scale_invariance(self):
        xyz = torch.randn(1, 64, 3)
        a = compute_stopo(xyz, K=12)
        b = compute_stopo(xyz * 3.0, K=12)
        self.assertTrue(torch.allclose(a, b, atol=1e-4))

    def test_chunked_gate(self):
        xyz = torch.randn(1, 64, 3)
        a = compute_stopo(xyz, K=12)
        b = compute_stopo_chunked(xyz, K=12, chunk_size=16, max_full_points=16)
        self.assertTrue(torch.allclose(a, b, atol=1e-4))

    def test_loss_has_finite_gradients(self):
        n = 24
        source = torch.randn(1, n, 16, requires_grad=True)
        target = torch.randn(1, n, 16, requires_grad=True)
        xyz = torch.randn(1, n, 3)
        projector = GSPProjector(num_superpoints=8)
        delta, _ = projector(xyz, F.normalize(source, dim=-1))
        gate = torch.linspace(0.0, 1.0, n).unsqueeze(0)
        index = torch.arange(n)
        objective = SoftClassifiedLoss(lambda_deg=0.05, lambda_var=200.0)
        loss, _ = objective(source, target, xyz, xyz, index, index, delta, gate)
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        self.assertTrue(torch.isfinite(source.grad).all())
        self.assertTrue(torch.isfinite(target.grad).all())


if __name__ == "__main__":
    unittest.main()
