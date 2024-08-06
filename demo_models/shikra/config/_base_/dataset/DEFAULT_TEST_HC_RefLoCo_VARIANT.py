REC_TEST_COMMON_CFG = dict(
    type='HCRefLoCoDataset',
    template_file=r'{{fileDirname}}/template/REC.json',
    dataset_path=r'../../HC-RefLoCo',
    split='val',
    max_dynamic_size=None,
)

DEFAULT_TEST_HC_RefLoCo = dict(
    REC_REF_HC_RefLoCo=dict(
        **REC_TEST_COMMON_CFG,
    )
)
