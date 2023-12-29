from time import time as tm
import faiss, hnswlib

def timer(func):
  '''
  декоратор, замеряющий время работы функции
  '''
  def wrapper(*args, **kwargs):
    start_time = tm()
    result = func(*args, **kwargs)
    end_time = tm() - start_time
    if isinstance(result, tuple):
      return *result, end_time
    return result, end_time
  return wrapper

@timer
def build_IVFFlat(build_data, **fixed_params):
  '''
  инициализация faiss.IndexIVF и сторирование build_data в нем
  returns index, build_time
  '''
  dim = fixed_params['dim']
  coarse_index = fixed_params['coarse_index']
  nlist = fixed_params['nlist']
  metric = fixed_params['metric'] if fixed_params['metric']!='cosine' else faiss.METRIC_INNER_PRODUCT
  
  num_threads = fixed_params.get('num_threads', 1)
  faiss.omp_set_num_threads(num_threads)
  
  index = faiss.IndexIVFFlat( # у faiss туго с именованными аргументами
    coarse_index, # индекс для поиска соседей-центроидов
    dim, # размерность исходных векторов
    nlist, # количество coarse-центроидов = ячеек таблицы
    metric # метрика, по которой считается расстояние между остатком(q) и [центроидом остатка](x)
  )
  newbuild_data = build_data.copy()
  if fixed_params['metric']=='cosine': faiss.normalize_L2(newbuild_data)
  index.train(newbuild_data)
  index.add(newbuild_data)
  return index # из-за декоратора ожидайте, что возвращается index, build_time

@timer
def build_IVFPQ(build_data, **fixed_params):
  '''
  инициализация faiss.IndexIVFPQ и сторирование build_data в нем
  returns index, build_time
  '''
  dim = fixed_params['dim']
  coarse_index = fixed_params['coarse_index']
  nlist = fixed_params['nlist']
  m = fixed_params['m']
  nbits = fixed_params['nbits']
  metric = fixed_params['metric']
  
  num_threads = fixed_params.get('num_threads', 1)
  faiss.omp_set_num_threads(num_threads)
  
  index = faiss.IndexIVFPQ( # у faiss туго с именованными аргументами
    coarse_index, # индекс для поиска соседей-центроидов
    dim, # размерность исходных векторов
    nlist, # количество coarse-центроидов = ячеек таблицы
    m, # на какое кол-во подвекторов бить исходные для PQ
    nbits, # log2 k* - количество бит на один маленький (составной) PQ-центроид
    metric # метрика, по которой считается расстояние между остатком(q) и [pq-центроидом остатка](x)
  )
  index.train(build_data)
  index.add(build_data)
  return index # из-за декоратора ожидайте, что возвращается index, build_time

@timer
def search_faiss(index, query_data, k, **kwargs):
  '''
  поиск среди ближайших соседей-центроидов для query_data
  returns distances, labels, search_time
  '''
  if 'nprobe' in kwargs:
    index.nprobe = kwargs['nprobe'] # количество ячеек таблицы, в которые мы заглядываем. Мы заглядываем в nprobe ближайших coarse-центроидов для q
  newquery_data = query_data.copy()
  if 'isCosine' in kwargs and kwargs['isCosine']: faiss.normalize_L2(newquery_data)
  distances, labels = index.search(newquery_data, k)
  return distances, labels # из-за декоратора ожидайте, что возвращается distances, labels, search_time

@timer
def build_hnsw(build_data, **fixed_params):
  '''
  инициализация hnswlib.Index и сторирование build_data в нем
  returns index, build_time
  '''
  dim = fixed_params['dim'] # размерность исходных векторов
  ef_construction = fixed_params['ef_construction'] # defines a construction time/accuracy trade-off
  M = fixed_params['M'] # defines tha maximum number of outgoing connections in the graph
  num_threads = fixed_params.get('num_threads', 1)
  space = fixed_params['space'] # possible options are l2, cosine or ip
  index = hnswlib.Index(space=space, dim=dim)
  index.init_index(max_elements = build_data.shape[0], ef_construction=ef_construction, M=M)
  index.add_items(build_data, num_threads = num_threads)
  return index

@timer
def search_hnsw(index, query_data, k, **kwargs):
  '''
  поиск ближайших k соседей обученного на build_data hnswlib.Index для query_data 
  returns distances, labels, search_time
  '''
  index.set_ef(kwargs['efSearch']) # чем больше тем качественнее соседи
  labels, distances = index.knn_query(query_data, k=k)
  return distances, labels

@timer
def build_flat_l2(build_data, dim):
  '''
  инициализация плоского(полного) индекса и сторирование build_data в нем
  returns index, build_time
  '''
  index = faiss.IndexFlatL2(dim)
  index.add(build_data)
  return index

@timer
def build_flat_cosine(build_data, dim):
  '''
  инициализация плоского(полного) индекса и сторирование build_data в нем
  returns index, build_time
  '''
  index = faiss.IndexFlatIP(dim)
  newbuild_data = build_data.copy()
  faiss.normalize_L2(newbuild_data)
  index.add(newbuild_data)
  return index

@timer
def search_flat_cosine(index, query_data, k):
  '''
  инициализация плоского(полного) индекса и сторирование build_data в нем
  returns index, build_time
  '''
  newquery_data = query_data.copy()
  faiss.normalize_L2(newquery_data)
  distances, labels = index.search(newquery_data, k)
  return distances, labels


@timer
def search_flat(index, query_data, k):
  '''
  поиск ближайших k соседей с помощью плоского индекса
  returns distances, labels, search_time
  '''
  distances, labels = index.search(query_data, k)
  return distances, labels
