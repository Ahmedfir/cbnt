class JobConfig:

    def __init__(self, add_cosine=True, add_cosine_nosuff=True, add_match_orig_nosuff=True, memory_aware=True,
                 cosine_func='torch'):
        self.add_cosine = add_cosine
        self.add_cosine_nosuff = add_cosine_nosuff
        self.add_match_orig_nosuff = add_match_orig_nosuff
        self.memory_aware = memory_aware
        self.cosine_func = cosine_func


DEFAULT_JOB_CONFIG = JobConfig()
NOCOSINE_JOB_CONFIG = JobConfig(add_cosine=False, add_cosine_nosuff=False)
