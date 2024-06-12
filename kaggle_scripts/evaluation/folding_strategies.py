from sklearn.model_selection import StratifiedKFold, KFold, GroupKFold


def stratified_k_fold(data, n_splits, random_state):
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    data["fold"] = -1
    for i, (_, val_index) in enumerate(skf.split(data, data["score"])):
        data.loc[val_index, "fold"] = i
    return data


def k_fold(data, n_splits, random_state):
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    data["fold"] = -1
    for i, (_, val_index) in enumerate(kf.split(data)):
        data.loc[val_index, "fold"] = i
    return data


def group_k_fold(data, n_splits, random_state):
    gkf = GroupKFold(n_splits=n_splits)
    data["fold"] = -1
    for i, (_, val_index) in enumerate(gkf.split(data, groups=data['group'])):
        data.loc[val_index, "fold"] = i
    return data


# Function to create folds based on the selected strategy
def create_folds(data, strategy, n_splits, random_state, groups=None):
    if strategy == "StratifiedKFold":
        return stratified_k_fold(data, n_splits, random_state)
    elif strategy == "KFold":
        return k_fold(data, n_splits, random_state)
    elif strategy == "GroupKFold":
        if groups is None:
            raise ValueError("Groups must be provided for GroupKFold")
        return group_k_fold(data, n_splits, random_state)
    else:
        raise ValueError(f"Unknown folding strategy: {strategy}")
