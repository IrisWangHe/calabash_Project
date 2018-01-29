import random
import numpy as np
import math
def power_by_mtt(graph):
    """Calculate the total power of the state, by the matrix-tree theorem.
    """
    #check=np.zeros((n+1,n+1),dtype=np.float64)
        #for u,v,w in check_edges:
        #if (u==0 or u in state) and v in state:
        #   check[abs(u),abs(v)]=w

    #mat_l = np.zeros((n+1, n+1), dtype=np.float64)
    matrix=np.diag(np.sum(graph,axis=0))-graph
    det = np.linalg.det(matrix[1:, 1:])
    return det
'''
    for i in range(n+1):
        for j in range(n+1):
            if i == j:
                mal_l

                for k in range(n+1):
                    if k != i:
                        mat_l[i, j] += graph[k, i]
            else:
                mat_l[i, j] = -graph[i, j]
'''

def find(i,j,edges):
    if i>=0:
        if j>0:
            w=edges[i,j]
        else:
            w=edges[i,abs(j)+n]
    else:
        if j>0:
            w=edges[abs(i)+n,j]
        else:
            w=edges[abs(i)+n,abs(j)+n]
    return w


def randomized_algorithm(n,edges,ind,init_state):

    #1:(3,10000)
    #2:(22,30000)
    #2:have tried:43,52


    times = 70000
    random.seed(ind)
    state=[]
    best_state=[]
    best_power=-1
    zero_time=0
        #for i in range(1, n+1):
        #state.append(i * (-1)**random.randrange(1, 3))
    
    state=init_state
    n = len(state)
    graph = np.zeros((n+1, n+1), dtype=np.float64)
    for i in state:
        for j in state: 
            if j!=i:
                w=find(i,j,edges)
                graph[abs(i),abs(j)]=w
    for i in state:
        w=find(0,i,edges)
        graph[0,abs(i)]=w


    count=0
    kk=0
    for _ in range(times):
        change=random.choice(state)
        origin=graph
        for i in state:
            if i!=change:
                w1=find(-change,i,edges)
                graph[abs(change),abs(i)]=w1
                w2=find(i,-change,edges)
                graph[abs(i),abs(change)]=w2
                w3=find(0,-change,edges)
                graph[0,abs(change)]=w3



        
        pos=abs(change)
        state[pos-1]=-change
        
        power = power_by_mtt(graph)
        if power==0:
            kk+=1
        if best_state ==[] or best_power <= power:
            best_state = state
            best_power = power
            count=0
            if best_power>0:
                ans=pow(10,9)+ pow(10,6)*(math.log(best_power)-n*math.log(1.5))
                #print(ans)
        
        else:
            state[pos-1]=change
            for i in state:
                if i!=change:
                    w1=find(change,i,edges)
                    graph[abs(change),abs(i)]=w1
                    w2=find(i,change,edges)
                    graph[abs(i),abs(change)]=w2
                    w3=find(0,change,edges)
                    graph[0,abs(change)]=w3
            

            count+=1
        if count>1000:
            break
        if kk>1000:
            return [ind,0]
    assert best_state is not None
    print ' '.join('%+d' % i for i in best_state)
    return [ind,ans]



def read_input():
    def one_edge():
        line = raw_input()
        u, v, w = line.split()
        return int(u), int(v), 1.5*float(w)
    n = int(raw_input())
    edges = np.zeros((2*n+1, n*2+1), dtype=np.float64)
    for i in range(4*n**2-2*n):
        u,v,w=one_edge()
        #check_edges.append((u,v,w))
        
        if u>=0:
            if v>0:
                edges[u,v]=w
            else:
                edges[u,abs(v)+n]=w

        else:
            if v>0:
                edges[abs(u)+n,v]=w
            else:
                edges[abs(u)+n,abs(v)+n]=w
    return n, edges

def greedy_init(n,edges):
    init_state=[]
    for i in range(1,n+1):
        positive=0
        negative=0
        for j in range(n*2+1):
            positive+=edges[i,j]
            negative+=edges[i+n,j]
        if positive>negative:
            init_state.append(i)
        else:
            init_state.append(-i)
    return init_state

if __name__ == '__main__':
    n, edges= read_input()
    #print(n)
    init_state=greedy_init(n,edges)
    tempans=randomized_algorithm(n,edges,20,init_state)
    #print(tempans)
    #max=-1
    #7:1
'''
    for m in range(20,10000):
        tempans=randomized_algorithm(n,edges,m,init_state)
        if max<=tempans[1]:
            print(tempans)
            max=tempans[1]
'''
