import AnalysisDataset as ds

##############################################################################
# [ENTRYPOINTS]
##############################################################################
if __name__ == "__main__":
    print()
    print('##############################################################################')
    print(' [DATASET]')
    print('##############################################################################')
    data = ds.AnalysisDataset(ds.AnalysisDataset.DATA_PROJECTS_FILE_H5)
    print('##############################################################################')
    print()
    print('##############################################################################')
    print(' [ANALYSIS]')
    print('##############################################################################')
    data.print_statistics()
    # data.build_plots_numerical()
    data.build_plots_categorial()
    print('##############################################################################')