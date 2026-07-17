from train_test import run_10_fold_cv
from utils import print_10_fold_res, test_model_parameters
from case_study import case_study_test


if __name__ == "__main__":
    # model training
    # run_10_fold_cv()

    # print the metrics
    print_10_fold_res(base_dir="./res/model", flage=True)

    # case study
    # T2DM id: 192
    # malaria id: 664
    # TNBC id: 208
    # case_study_test(case_disease_id=192)