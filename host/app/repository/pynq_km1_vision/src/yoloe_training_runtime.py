"""Process-local training compatibility; does not edit installed libraries."""
import functools


def install_cpu_target_assignment():
    """Run the original no-gradient target matcher on CPU, then restore device.

    Model forward, trainable projections and optimizer stay on their requested
    device. All matcher inputs retain their original dtype and the upstream
    algorithm is used without changing thresholds or labels.
    """
    import torch
    from ultralytics.utils.tal import TaskAlignedAssigner
    if getattr(TaskAlignedAssigner.forward,'_fpga_cpu_assignment',False):return
    original=TaskAlignedAssigner.forward

    @functools.wraps(original)
    @torch.no_grad()
    def forward(self,*args,**kwargs):
        device=args[0].device if args else kwargs['pd_scores'].device
        def cpu(value):return value.detach().cpu() if isinstance(value,torch.Tensor) else value
        if device.type=='cpu':return original(self,*args,**kwargs)
        values=original(self,*(cpu(v) for v in args),**{k:cpu(v) for k,v in kwargs.items()})
        return tuple(v.to(device) for v in values)
    forward._fpga_cpu_assignment=True
    TaskAlignedAssigner.forward=forward
