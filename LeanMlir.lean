import LeanMlir.Types
import LeanMlir.Spec
import LeanMlir.F32Array
import LeanMlir.IreeRuntime
import LeanMlir.MlirCodegen
import LeanMlir.SpecHelpers
import LeanMlir.Train
-- VJP proofs (Attention pulls in Tensor/MLP/Residual/SE/LayerNorm/BatchNorm
-- transitively; CNN + Depthwise need explicit imports).
import LeanMlir.Proofs.Attention
import LeanMlir.Proofs.CNN
import LeanMlir.Proofs.Depthwise
