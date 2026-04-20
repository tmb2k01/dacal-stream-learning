import numpy as np

def get_oob_bootstraps(population_size, oob_times, bootstrap_fraction=1, **bootstrap_params):
    if bootstrap_params is None or len(bootstrap_params) == 0:
        bootstrap_params = {"bootstrap_type": "classic"}
    if "bootstrap_type" not in bootstrap_params.keys():
        raise ValueError("No bootstrap type defined")
    if bootstrap_params["bootstrap_type"] == "classic":
        return get_oob_bootstraps__classic(population_size, oob_times, bootstrap_fraction=1, **{k:v for k,v in bootstrap_params.items() if k != "bootstrap_type"})
    elif bootstrap_params["bootstrap_type"] == "greedy_cover":
        return get_oob_bootstraps__classic(population_size, oob_times, bootstrap_fraction=1, **{k:v for k,v in bootstrap_params.items() if k != "bootstrap_type"})
    else:
        raise ValueError(f"Unknown type: {bootstrap_params['bootstrap_type']}")

def get_oob_bootstraps__classic(population_size, oob_times, bootstrap_fraction=1, n_max_size=None):
        n_max_size = n_max_size if n_max_size is not None else np.inf

        bootstraps, cur_size = [], 0
        count = np.zeros(shape=population_size, dtype=int)
        while True:
            new_bootstraps = np.random.choice(range(population_size), size=(min(oob_times,n_max_size-cur_size),int(bootstrap_fraction*population_size)), replace=True)
            if new_bootstraps.shape[0] == 0:
                return np.vstack(bootstraps)
            for i, bootstrap in enumerate(new_bootstraps):
                c_strap = np.ones(shape=population_size, dtype=bool)
                c_strap[bootstrap] = False
                count += c_strap
                if count.min() >= oob_times:
                    bootstraps.append(new_bootstraps[:i])
                    return np.vstack(bootstraps)
            bootstraps.append(new_bootstraps)
            cur_size += new_bootstraps.shape[0]

def get_oob_bootstraps__greedy_cover(population_size, oob_times, bootstrap_fraction=1, n_repititions=50):
        n_repititions = bootstrap_params["n_repititions"]
        
        S = np.random.choice(range(population_size), size=(n_repititions*oob_times,int(bootstrap_fraction*population_size)), replace=True)
        M = np.ones(shape=(S.shape[0],population_size))
        
        for i,s in enumerate(S):
            M[i][s] = 0
        
        s = greedy_cover(M, np.ones(M.shape[0]), oob_times)
        return S[s]

def greedy_cover(M, w, d, selected = None):
    M = M != 0
    n, m = M.shape  
    r = np.zeros(m)
    
    if selected is None:
        selected = np.zeros(n, bool)  
    assert selected.shape == (n,)

    while r.min() < d and not selected.all():
        gain = M[:, r < d].sum(axis=1) / w
        gain[selected] = -1
        i = gain.argmax()
        r[M[i]] += 1
        selected[i] = True
    return selected

get_oob_bootstraps(500, 100, bootstrap_type="classic", n_max_size=150)
