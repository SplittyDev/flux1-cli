class AppConfig:
    """
    Configuration for the application.
    """

    offload_cpu: bool
    force_cpu: bool

    def __init__(self, offload_cpu: bool, force_cpu: bool):
        self.offload_cpu = offload_cpu
        self.force_cpu = force_cpu
