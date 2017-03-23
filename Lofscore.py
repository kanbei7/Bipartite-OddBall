import networkx as nx
import math

MinPts=5

def euclidean_dist(x1,x2):
    assert (len(x1)==len(x2))
    ans=0
    for i in range(len(x1)):
        tmp=(x1[i]-x2[i])
        ans+=(tmp*tmp)
    return math.sqrt(ans)

def cal_dist(points):
    tmp=nx.Graph()
    lst=list(points.keys())
    for i in range(len(lst)):
        a=lst[i]
        j=i+1
        while j<len(lst):
            b=lst[j]
            tmp.add_edge(a,b,weight=euclidean_dist(points[a],points[b]))
            j+=1
    return tmp

def cal_k_dist(points,DIST,NMPTS,NMPTS_set):
    ans={}
    lst=list(points.keys())
    for pt in lst:
        edge_lst=[]
        for y in lst:
            if y==pt:
                continue
            edge_lst.append((y , DIST[pt][y]['weight']))
        tmp_lst=sorted(edge_lst, key=lambda x: x[1])
        ans[pt]=tmp_lst[MinPts][1]
        N_set=[]
        for i in range(len(tmp_lst)):
            if tmp_lst[i][1]<=ans[pt]:
                N_set.append(tmp_lst[i][0])
            else:
                break
        NMPTS[pt]=len(N_set)
        NMPTS_set[pt]=set(N_set)
    return ans

def cal_reach_dist(points,K_DIST,DIST):
    tmp={}
    lst=list(points.keys())
    for i in range(len(lst)):
        a=lst[i]
        tmp[a]={}
        for j in range(len(lst)):
            if i==j:
                continue
            b=lst[j]
            tmp[a][b]=max(K_DIST[b],DIST[a][b]['weight'])
    return tmp

def calc_LRD(points,NMPTS,REACH_DIST,NMPTS_set):
    tmp={}
    lst=list(points.keys())
    for i in lst:
        tmp_sum=0.0
        for j in NMPTS_set[i]:
            tmp_sum+=REACH_DIST[i][j]
        tmp[i]=1.0*NMPTS[i]/tmp_sum
    return tmp

def calc_LOF(points,LRD,NMPTS,NMPTS_set):
    ans={}
    lst=list(points.keys())
    for i in lst:
        tmp=0.0
        for j in NMPTS_set[i]:
            tmp+=LRD[j]
        tmp/=LRD[i]
        ans[i]=tmp/NMPTS[i]
    return ans

def get_EWPLpoints(df):
    ans={}
    for index,row in df.iterrows():
        x_log=math.log(float(row['Ego-net Edges']),10)
        y_log=math.log(float(row['Egonet Weight']),10)
        ans[row['Node']]=[x_log,y_log]
    return ans

def get_ELWPLpoints(df):
    ans={}
    for index,row in df.iterrows():
        y_log=math.log(float(row['Eigenvalue']),10)
        x_log=math.log(float(row['Egonet Weight']),10)
        ans[row['Node']]=[x_log,y_log]
    return ans

def add_LOF(p,df,feat_name):
    NMPTS={}
    NMPTS_set={}
    DIST=cal_dist(p)
    K_DIST=cal_k_dist(p,DIST,NMPTS,NMPTS_set)
    REACH_DIST=cal_reach_dist(p,K_DIST,DIST)
    LRD=calc_LRD(p,NMPTS,REACH_DIST,NMPTS_set)
    LOF=calc_LOF(p,LRD,NMPTS,NMPTS_set)

    ans=[]
    for index, row in df.iterrows():
        if row['Node'] in LOF:
            ans.append(float(LOF[row['Node']]))
        else:
            ans.append(0.0)
    df[feat_name]=ans

def add_weightLOF_feat(df):
    p=get_EWPLpoints(df)
    add_LOF(p,df,'weight_LOF')

def add_eigenLOF_feat(df):
    p1=get_ELWPLpoints(df)
    add_LOF(p1,df,'eigen_LOF')


