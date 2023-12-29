import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter



def calc_recall(true_labels, pred_labels, k, exclude_self=False, return_mistakes=False):
  '''
  счиатет recall@k для приближенного поиска соседей
  
  true_labels: np.array (n_samples, k)
  pred_labels: np.array (n_samples, k)
  
  exclude_self: bool
      Если query_data была в трейне, считаем recall по k ближайшим соседям, не считая самого себя
  return_mistakes: bool
      Возвращать ли ошибки
  
  returns:
      recall@k
      mistakes: np.array (n_samples, ) с количеством ошибок
  '''
  n = true_labels.shape[0]
  n_success = []
  shift = int(exclude_self)
  
  for i in range(n):
      n_success.append(np.intersect1d(true_labels[i, shift:k+shift], pred_labels[i, shift:k+shift]).shape[0])
      
  recall = sum(n_success) / n / k
  if return_mistakes:
      mistakes = k - np.array(n_success)
      return recall, mistakes
  return recall


def plot_ann_performance(*args, **kwargs):
  '''
  рисует сравнительный график времени работы и recall@k 
  для имплементированных методов (переданных в index_dict) приближенного поиска соседей
  
  kwargs:
  - build_data: np.array (n_samples, dim)
  - query_data: np.array (n_samples, dim)
  - flat_build_func: (build_data) -> (index, build_t)
      flat build knn function for comparison 
  - flat_search_func: (index, query_data, k) -> (distances, labels, search_t)
      flat search knn function for comparison
  - query_in_train: bool
      passed to calc_recall as exclude_self
  - title: str
  - qps_line: float
      plots horisontal line at this value
  - recall_line: float or list of floats
      plots vertical lines at these values
  - index_dict: dict,
      key: model_name
      value: dict
        fixed_params: dict for params used in build_func
        build_func: (build_data, fixed_params) -> (index, build_t)
        search_func: (index, query_data, k, search_param) -> (distances, labels, search_t)
        search_param: tuple (имя параметра поиска, [используемые значения])
  - k: int
      number of nearest neighbors for each query (passed to search_func)

  returns:
      None
  '''
  n = len(kwargs['index_dict'])
  _mypalette = ["#F72585", "#4361EE", "#3A0CA3", "#7209B7", "#4CC9F0"]
  palette = _mypalette if n<= 5 else sns.color_palette("husl", n)
  palette = palette[:n]

  build_data = kwargs['build_data']
  query_data = kwargs['query_data']
  k          = kwargs['k']

  # build-search flat, no need in build_t and distances
  index, _ = kwargs['flat_build_func'](build_data)
  _, flat_labels, flat_search_t = kwargs['flat_search_func'](index, query_data, k)

  index_info = {}
  # build-search index:
  for model_name in kwargs['index_dict']:
    # build:
    build_f     = kwargs['index_dict'][model_name]['build_func']
    build_param = kwargs['index_dict'][model_name]['fixed_params']
    index, build_t = build_f(build_data, **build_param)

    # search:
    search_f = kwargs['index_dict'][model_name]['search_func']
    search_param_name, search_param = kwargs['index_dict'][model_name]['search_param']
    index_info[model_name] = {
       'build_t': build_t, 'search_param_name': (search_param_name, search_param),
       'recall': [], 'qps': []
       }
    for param in search_param:
      _, labels, search_t = search_f(index, query_data, k, **{search_param_name: param, 'isCosine': ('metric' in build_param) and (build_param['metric'] == 'cosine')})
      #recall:
      recall = calc_recall(flat_labels, labels, k, exclude_self=kwargs['query_in_train'])
      index_info[model_name]['recall']   += [recall]
      index_info[model_name]['qps'] += [len(query_data)/search_t]
  del index

  # build-time
  fig, ax = plt.subplot_mosaic('ABB;.BB', figsize=(14, 7))
  build_ts = [index_info[model_name]['build_t'] for model_name in index_info]
  labels = list(index_info.keys())
  sns.barplot(x=labels, y=build_ts, hue=labels, palette=palette, ax=ax['A'], legend=False, color='black')
  ax['A'].tick_params('x', rotation=90, labelsize=14)
  ax['A'].tick_params('y', labelsize=14)
  ax['A'].set_title('build time',  fontsize=14)
  ax['A'].set_ylabel('time (s)',  fontsize=14)
  ax['A'].grid(axis='x', alpha=0.5)
  ax['A'].grid(axis='y', alpha=0.5)

  # search-time
  for i, model_name in enumerate(index_info):
    if len(index_info[model_name]['search_param_name'][1]) == 1:
      ax['B'].plot(index_info[model_name]['recall'], index_info[model_name]['qps'], '*', color=palette[i], label=model_name, markersize=14)
      continue
    ax['B'].plot(
      index_info[model_name]['recall'],
      index_info[model_name]['qps'],
      '*-',
      label=model_name,
      color=palette[i],
      markersize=12
    )
    search_param = index_info[model_name]['search_param_name']
    # annotate all dots with search_param_name=param:
    for j, param in enumerate(search_param[1]):
      ax['B'].annotate(
         f'{search_param[0]}={param}', 
         (
            index_info[model_name]['recall'][j], 
            index_info[model_name]['qps'][j],
            ),
         )
    
  # extra
  ax['B'].axhline(y=len(query_data)/flat_search_t, color='r', linestyle='--', 
                  label=f'flat:'+'{:0.3e}'.format(flat_search_t)
                  )
  if 'qps_line' in kwargs:
    ax['B'].axhline(y=kwargs['qps_line']*len(query_data)/flat_search_t, color='darkgreen', linestyle='--')
  if 'recall_line' in kwargs:
    # check if kwargs['recall_line'] is int:
    if isinstance(kwargs['recall_line'], int):
      ax['B'].axvline(x=kwargs['recall_line'], color='darkgreen', linestyle='--')
    else: # it is list
      for r in kwargs['recall_line']:
        ax['B'].axvline(x=r, color='darkgreen', linestyle='--')
      
  ax['B'].set_title(kwargs.get('title', 'recall-time'), fontsize=16)
  ax['B'].legend()
  ax['B'].grid(axis='y', alpha=0.5)
  ax['B'].set_xlabel(f'recall@{k}', fontsize=14)
  ax['B'].set_ylabel('queries per second', fontsize=14)
  ax['B'].set_yscale('log')
  ax['B'].tick_params('y', labelsize=14)
  ax['B'].tick_params('x', labelsize=14)

  fig.tight_layout()
  plt.show()

    
def analyze_ann_method(*args, **kwargs):
  '''
  рисует распределение ошибок переданного метода приближенного knn
  по оси x: #ошибок (от 0 до k) per 1 запрос
  по оси y: сколько раз было получено x ошибок среди всех запросов

  kwargs:
    те же что и в plot_ann_method
  - build_data
  - query_data
  - k
  - build_func
  - search_func
  - flat_build_func
  - flat_search_func
  - query_in_train
  - index_name
    то же что и title
  '''
  build_data = kwargs['build_data']
  query_data = kwargs['query_data']
  k          = kwargs['k']
  build_func = kwargs['build_func']
  search_func = kwargs['search_func']
  flat_build_func  = kwargs['flat_build_func']
  flat_search_func = kwargs['flat_search_func']
  query_in_train   = kwargs['query_in_train']
  index_name = kwargs['index_name']

  index, _ = flat_build_func(build_data)
  _, flat_labels, _ = flat_search_func(index, query_data, k)

  index, build_t = build_func(build_data)
  _, labels, search_t = search_func(index, query_data, k)

  recall, mistakes = calc_recall(flat_labels, labels, k, exclude_self=query_in_train, return_mistakes=True)
  
  count = Counter(mistakes)
  count = {i: count.get(i, 0) for i in range(k+1)}
  df = pd.DataFrame(count.items(), columns=['mistakes', 'count'])
  df['count'] = df['count'].replace(0, -.1*max(count.values()))
  fig = plt.figure(figsize=(10, 6))
  label = f'build time: {build_t:.3f} s\nsearch time: {search_t:.3f} s\nrecall@{k}: {recall:.3f}'
  ax = sns.barplot(data=df, x='mistakes', y='count', edgecolor='black', color='#4361EE', label=label)
  values = [max(int(i.get_height()), 0) for i in ax.containers[0]]
  ax.bar_label(ax.containers[0], labels=values, fontsize=13, color='firebrick')
  plt.ylim(-.25*max(count.values()), 1.1*max(count.values()))
  plt.legend()
  plt.setp(ax.get_legend().get_texts(), fontsize='22')
  plt.axhline(0, color='black', ls='--')
  plt.title(index_name, fontsize=16)
  plt.xlabel(ax.get_xlabel(), fontsize=14)
  plt.ylabel(ax.get_ylabel(), fontsize=14)
  plt.tick_params('x', labelsize=14)
  plt.tick_params('y', labelsize=14)
  plt.show()




# Для FASHION MNIST
def knn_predict_classification(neighbor_ids, tr_labels, n_classes, distances=None, weights='uniform'):
  '''
  по расстояниям и айдишникам получает ответ для задачи классификации
  
  distances: (n_samples, k) - расстояния до соседей
  neighbor_ids: (n_samples, k) - айдишники соседей
  tr_labels: (n_samples,) - метки трейна
  n_classes: кол-во классов
  
  returns:
      labels: (n_samples,) - предсказанные метки
  '''
  
  n, k = neighbor_ids.shape

  labels = np.take(tr_labels, neighbor_ids)
  labels = np.add(labels, np.arange(n).reshape(-1, 1) * n_classes, out=labels)

  if weights == 'uniform':
      w = np.ones(n * k)
  elif weights == 'distance' and distances is not None:
      w = 1. / (distances.ravel() + 1e-10)
  else:
      raise NotImplementedError()
      
  labels = np.bincount(labels.ravel(), weights=w, minlength=n * n_classes)
  labels = labels.reshape(n, n_classes).argmax(axis=1).ravel()
  return labels


# Для крабов!
def get_k_neighbors(distances, k):
  '''
  считает по матрице попарных расстояний метки k ближайших соседей
  
  distances: (n_queries, n_samples)
  k: кол-во соседей
  
  returns:
      labels: (n_queries, k) - метки соседей
  '''
  indices = np.argpartition(distances, k - 1, axis=1)[:, :k]
  lowest_distances = np.take_along_axis(distances, indices, axis=1)
  neighbors_idx = lowest_distances.argsort(axis=1)
  indices = np.take_along_axis(indices, neighbors_idx, axis=1) # sorted
  sorted_distances = np.take_along_axis(distances, indices, axis=1)
  return sorted_distances, indices


# Для крабов! Пишите сами...
def knn_predict_regression(labels, y, weights='uniform', distances=None):
  '''
  по расстояниям и айдишникам получает ответ для задачи регрессии
  я просто использовал knn.predict(
  '''
  pass
