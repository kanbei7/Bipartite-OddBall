import re
import math
import time
import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import Lofscore as Lf

#============Graph===============#
class EgonetFeatures:
	nodeId = None
	egoNetGraph = None
	egoNetDegree = 0
	nofEdges = 0
	totalWeight = 0
	eigenValue = 0.0
	entropy = -1.0
	def get_properties_sequence(self):
		return [self.nodeId, str(nx.nodes(self.egoNetGraph)), str(self.egoNetDegree) , \
		str(self.nofEdges) , str(self.totalWeight), str(self.eigenValue), str(self.entropy)]

	def __str__(self):
		return "Node: " + self.nodeId + "Egonet" + str(nx.nodes(self.egoNetGraph)) + \
		", Degree: " + str(self.egoNetDegree) + " , Edges: " + str(self.nofEdges) + "Weight: " + str(self.totalWeight) + "Eigenvalue"+ str(self.eigenValue)+ "Entropy"+ str(self.entropy)

#--------End of Class Definition--------#

def weight_egonet(g):
	wgt = float(sum(g.degree(weight='weight').values()))/float(2)
	return wgt

def weight_io(g,n):
	temp = []
	n1 = nx.ego_graph(g, n, radius=1, center=True, undirected=False, distance=None)
	n1Edgs = n1.edges()
	for v in nx.all_neighbors(g,n):
		n2 = nx.ego_graph(g, v, radius=1, center=True, undirected=False, distance=None)
		n2Edgs = n2.edges()
		n3Edgs = list(set(n2Edgs) - set(n1Edgs))
		temp2=[]
		for e in n3Edgs:
			wgt = g.get_edge_data(e[0], e[1])
			w = wgt.get('weight')
			temp2.append(w)
		temp.append(np.sum(temp2))
	return np.sum(float(temp))

def adj_matrix_eigen(g,n):
	n1 = nx.ego_graph(g, n, radius=1, center=True, undirected=False, distance=None)
	L = nx.adjacency_matrix(n1, weight='weight')
	e = np.linalg.eigvals(L.A)
	PrincipalEig = max(e)
	return PrincipalEig.real

def ego_entropy(g,n):
	tmp_entropy=0.0
	wsum=1.0*weight_egonet(g)
	for e in g.edges_iter(data=True):
		if e[2]['weight']!=0:
			pk=e[2]['weight']/wsum
			tmp_entropy+=pk*math.log(1/pk,10)
	return tmp_entropy/math.log(nx.number_of_nodes(g),10)

def create_graph_features_all(g):
	assert type(g) is not type(None)
	egoNetList = []
	for n in nx.nodes_iter(g):
		resultObj = EgonetFeatures()
		resultObj.nodeId = n
		resultObj.egoNetGraph = nx.ego_graph(g, n, radius=1, center=True)
		resultObj.egoNetDegree = nx.number_of_nodes(resultObj.egoNetGraph)
		resultObj.nofEdges = nx.number_of_edges(resultObj.egoNetGraph)
		resultObj.totalWeight = weight_egonet(resultObj.egoNetGraph)
		resultObj.eigenValue = adj_matrix_eigen(g,n)
		resultObj.entropy = ego_entropy(resultObj.egoNetGraph,n)
		egoNetList.append(resultObj)
	return egoNetList

def write_results_to_dataframe_all(results):
	result_one = []
	for item in results:
		seq = item.get_properties_sequence()
		result_one.append(seq)
	dataFrame = pd.DataFrame(result_one, columns=["Node", "Egonet", "Ego-net degree", \
		"Ego-net Edges", "Egonet Weight","Eigenvalue","Egonet Entropy"])
	return dataFrame


#Define methods for calculating weight
def totalCost(x):
	return float(x)

def simpleCount(x):
	if float(x)>0.0:
		return 1.0
	else:
		return 0.0


WEIGHT_METHODS={'total':totalCost ,'simple':simpleCount }
#compute weight for every edges according to wm. Input Args: dataframe, src column name, dst column name, \
#weight column name, range for src list, src list, weight method
def get_weight(data,x,y,w,srclst,a,b,wm='total'):
	dataSrc = srclst[a:b]
	Src = []
	Wgt = []
	Dst = []
	for i in dataSrc:
		DF1 = data[data[x] == i]
		dataDst = set(DF1[y])
		for j in dataDst:
			DF2 = DF1[DF1[y] == j]
			weight = 0.0
			for index,row in DF2.iterrows():
				weight+=WEIGHT_METHODS[wm](row[w])
			Src.append(i)
			Dst.append(j)
			Wgt.append(weight)
	dataWeight = zip(Src, Dst,Wgt)
	return dataWeight

def pickShorter(lst1,lst2):
	if len(lst1)>len(lst2):
		return lst2
	else:
		return lst1

#given a dataframe, with nodes columns and edge columns specified
# !! return A LIST OF CONNECTED_SUBGRAPHs (for computational efficiency)
def createGraph(data, x, y, wgt,wm):
	print ("Start Creating Graph...")
	X_lst=list(set(data[x]))
	Y_lst=list(set(data[y]))
	print ("Number of "+x+": "+str(len(X_lst)))
	print ("Number of "+y+": "+str(len(Y_lst)))
	input_lst=pickShorter(X_lst,Y_lst)
	edges = get_weight(data, x, y, wgt, input_lst, 0, len(input_lst), wm)
	print ("Edges Created.")

	g = nx.MultiGraph()
	g.add_weighted_edges_from(edges)
	graphs=list(nx.connected_component_subgraphs(g))
	print ("Graph created!")
	print ("Number of Edges:"+str(len(g.edges())))
	print ("Number of Connected Subgraphs:"+str(len(graphs)))

	return graphs

#
def createFeatures(graphs):
	print ("Start Creating Features...")
	tmp_frames=[]
	for gi in graphs:
		print (str(len(gi.nodes())))
	for gi in graphs:
		resultList = create_graph_features_all(gi)
		tmp_frames.append(write_results_to_dataframe_all(resultList))
		print ('check')
	Features=pd.concat(tmp_frames)
	print ("Egonet Features Created!")
	print ("Number of Egonets:"+str(len(Features)))
	return Features

#============Outlier Scores===============#
def anomalyScore(y,y_expect):
	return math.log((abs(y-y_expect)+1),10)

def ADDTAG(df,tag):
	df['node_class']=[tag]*len(df)

def EWPL(df):
	score=[]
	x=[]
	y=[]
	x_log=[]
	y_log=[]
	nodes=[]
	for index,row in df.iterrows():
		if float(row['Egonet Weight'])>0:
			x_log.append(math.log(float(row['Ego-net Edges']),10))
			y_log.append(math.log(float(row['Egonet Weight']),10))
			x.append(float(row['Ego-net Edges']))
			y.append(float(row['Egonet Weight']))
			nodes.append(row['Node'])
	x_log=np.array(x_log)
	y_log=np.array(y_log)
	x=np.array(x)
	y=np.array(y)
	z=np.polyfit(x_log, y_log, 1)
	p = np.poly1d(z)
	result={}
	for i in range(len(x)):
		result[nodes[i]]=anomalyScore(y[i],math.pow(10,p(math.log(x[i],10))))
	ans=[]
	for index,row in df.iterrows():
		if row['Node'] in result:
			ans.append(result[row['Node']])
		else:
			ans.append(np.NaN)
	df['weight_outlier']=ans
	
def ELWPL(df):
	score=[]
	x=[]
	y=[]
	x_log=[]
	y_log=[]
	nodes=[]
	for index,row in df.iterrows():
		if float(row['Egonet Weight'])>0 and float(row['Eigenvalue']):
			y_log.append(math.log(float(row['Eigenvalue']),10))
			x_log.append(math.log(float(row['Egonet Weight']),10))  
			y.append(float(row['Eigenvalue']))
			x.append(float(row['Egonet Weight']))
			nodes.append(row['Node'])
	x_log=np.array(x_log)
	y_log=np.array(y_log)
	x=np.array(x)
	y=np.array(y)
	z=np.polyfit(x_log, y_log, 1)
	p = np.poly1d(z)
	result={}
	for i in range(len(x)):
		result[nodes[i]]=anomalyScore(y[i],math.pow(10,p(math.log(x[i],10))))
	ans=[]
	for index,row in df.iterrows():
		if row['Node'] in result:
			ans.append(result[row['Node']])
		else:
			ans.append(np.NaN)
	df['eigen_outlier']=ans

#metric={wpl,eigpl,entropy}
def calc_score(df,x,y,lst1,metrics):
	frame=[]
	for index ,row in df.iterrows():
		if row['Node'] in lst1:
			frame.append(row)
	Feat=pd.DataFrame(frame)
	
	if metrics=='wpl':
		EWPL(Feat)
		Feat=Feat.sort_values(by=['weight_outlier'],ascending=[0])
		Lf.add_weightLOF_feat(Feat)

		wlof_max=max(Feat.weight_LOF)
		print ("Max LOF Score: "+str(wlof_max))
		wpl_max=max(Feat.weight_outlier)
		print ("Max Weight Outlier Score: "+str(wpl_max))
		#normalization
		res=[]
		for index,row in Feat.iterrows():
			res.append(row['weight_outlier']/wpl_max+row['weight_LOF']/wlof_max)
		Feat['outlier_score']=res
		return Feat.sort_values(by=['outlier_score'],ascending=[0])

	elif metrics=='eigpl':
		ELWPL(Feat)
		Feat=Feat.sort_values(by=['eigen_outlier'],ascending=[0])
		Lf.add_eigenLOF_feat(Feat)

		elof_max=max(Feat.eigen_LOF)
		print ("Max LOF Score: "+str(elof_max))
		epl_max=max(Feat.eigen_outlier)
		print ("Max Eigen Outlier Score: "+str(epl_max))
		#normalization
		res=[]
		for index,row in Feat.iterrows():
			res.append(row['eigen_outlier']/epl_max+row['eigen_LOF']/elof_max)
		Feat['outlier_score']=res
		return Feat.sort_values(by=['outlier_score'],ascending=[0])
		
	elif metrics=='entropy':
		return Feat.sort_values(by=['Egonet Entropy','Egonet Weight'],ascending=[1,0])

def write_score(result,filename):
	result.to_csv(filename)

