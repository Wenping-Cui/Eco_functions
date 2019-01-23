import numpy as np

##########################################################################
# metabolic_flag="random", "fixed", "tiled"
def Make_D_matrices(S, M, nu, p, q, energies, flag):
    """M- number of resources; nu- highest trophic layer for any species; p-probability of leakage; 
       q-prob of adding pathway; There is always
    atleast one species at trophic level nu rest of nu (top trophic layer) are randomly drawn between nu and M;
    """
    # Create master D matrix for vu=0 and q=1 for fixed p
    metabolic_flag = flag[0]
    community_flag =flag[1]
    delta = lambda x, y: 1 if x == y else 0;
    nu = np.minimum(M - 1, nu);
    if metabolic_flag == "random":
        nu_array = np.append(np.array([nu]),np.random.randint(nu,M-1,size=S-1))

    if metabolic_flag == "fixed":
        nu_array = nu * np.ones(S)

    if metabolic_flag == "tiled":
        nu_array = np.mod(np.arange(nu, S + 1) - nu, M - nu) + nu

    if metabolic_flag == "one-step":
        nu_array = nu * np.ones(S)
    ecosystem = []
    for j in range(0, S):
        nu = nu_array[j]
        D_matrix = np.zeros((M, M));

        for a in xrange(M):
            if np.random.rand()<=q:
                    D_matrix[a,:] = 1.0
                    if np.random.rand()< 0.95:
                           if community_flag=='A' and a%2==1:
                                D_matrix[a,:] = 0.0 
                           if community_flag=='B' and a%2==0:
                                D_matrix[a,:] = 0.0
                                 
        if community_flag=='A':
                 D_matrix[M-2,:] = 1.0 
                 D_matrix[M-1,:] = 0 
        if community_flag=='B':
                 D_matrix[M-1,:] = 1.0  
                 D_matrix[M-2,:] = 0                  
        for a in range(0, M):
            for b in range(0, M):
                if a <= b or b < nu:
                    D_matrix[a, b] = 0
      #  D_matrix[M-1,:]=0            
                #else:
                #    if np.random.rand()<=q:
                #        D_matrix[a, b] = 1.0         
        for i in xrange(M):
            D_vec = np.copy(D_matrix[:,i])  
            Nozero_indice=np.nonzero(D_vec)[0] 
            if len(Nozero_indice)>0:    
                r = np.random.rand(len(Nozero_indice));
                r = r/np.sum(r)*p;
                for D_i in xrange(len(Nozero_indice)):
                    indice = Nozero_indice[D_i]
                    D_vec[indice] = r[D_i]/energies[indice]*energies[i]
            D_matrix[:,i]=D_vec[:]
        D_matrix[np.where(D_matrix < 10 ** -5)] = 0;
        ecosystem.append(D_matrix)
    if metabolic_flag == "one-step":
        ecosystem1 = [];
        for j in range(M-1):
            D = np.zeros((M, M))
            D[j + 1, j] = 1
            ecosystem1.append(D)
    return ecosystem1 if metabolic_flag == "one-step" else ecosystem