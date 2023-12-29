import numpy as np
import pandas as pd
from tqdm import tqdm
from ipywidgets import widgets
from IPython.display import display

import plotly.express as px
import plotly.graph_objects as go

from bokeh.plotting import figure, show
from bokeh.models import ColumnDataSource, FixedTicker, BooleanFilter, CDSView, HoverTool
from bokeh.io import output_notebook
from bokeh.layouts import gridplot
from bokeh.transform import linear_cmap
from bokeh.palettes import magma, tol, Inferno


class DimReductionPlotter:
  def __init__(self):
    self.bokeh_first_time = True
        
  def get_results(self, data, mapper_dict, default_features=None):
    '''
    Функция принимает на вход данные и набор 2D/3D dimension-редукторов через mapper_dict.
    
    data - pd.DataFrame со всеми необходимыми данными - hue_cols, features

    mapper_dict - словарь знакомого вида :)

    default_features: array of strings - фичи которые будут использоваться для вычисления функции расстояния,
        если для reductor`а не указано иного
        
        
    returns
        results: dict

        Note: для t-SNE mapper=embedding и лежит по ключу 'embedding'!
            Это объект класса TSNEEmbedding, это "обертка" над эмбеддингом.
            У него есть метод transform, а также его можно воспринимать как эмбеддинг и, например, слайсить и рисовать
    '''
    results = dict()
    for mapper_name in tqdm(mapper_dict):
      mapper_props = mapper_dict[mapper_name]
      params, features = mapper_props['params'], mapper_props.get('features', default_features)
      if features is None:
        raise ValueError(f'Мапперу {mapper_name} нужно указать фичи')
      mapper, embedding, time_passed = mapper_props['func'](data[features].values, params)
      results[mapper_name] = {
        'embedding': embedding,
        'mapper': mapper
      }
    return results
     
  def plot_plotly_fig(self, embedding, hue, hue_name, title='', hover_data=None, cat_trsh=24, hue_cols=None):
    '''
    embedding: 3D array
    hue: data[hue_col] в нужном формате (строки, если категории)
    hover_data: pd.DataFrame с данными для hover
    cat_trsh: int - порог для категориальных признаков
    hue_cols: pd.DataFrame с данными для dropdown menu динамической смены hue
    '''
    x, y, z = embedding[:, 0], embedding[:, 1], embedding[:, 2]

    #! new
    plot_data = pd.concat([pd.DataFrame({'x': x, 'y': y, 'z': z, hue_name: hue}), hover_data], axis=1)
    plot_data = plot_data.loc[:,~plot_data.columns.duplicated()]
    hover_data = list(hover_data.keys()) if hover_data is not None else None

    #! new everything below
    plotly_fig = px.scatter_3d(plot_data, x='x', y='y', z='z', title=title, hover_data=hover_data)
    if hue_cols is None:
      plotly_fig = px.scatter_3d(plot_data, x='x', y='y', z='z', title=title, color=hue_name, hover_data=hover_data)
    else:
      def set_categorical_colors(column, cat_trsh):
        self.dropdown_col = column
        nunique = hue_cols[column].nunique()
        if nunique <= min(cat_trsh, 24):
          colors = px.colors.qualitative.D3[:nunique] if nunique <= 10 else px.colors.qualitative.Dark24[:nunique]
          category_mapping = {value: color for value, color in zip(hue_cols[column].unique(), colors)}
          return hue_cols[column].map(category_mapping)
        return hue_cols[column]
        
      # Add the hue button
      hue_menu = [
        dict(
          label=column,
          method='restyle',
          args=[{'marker.color': [set_categorical_colors(column, cat_trsh)]}],
          )
        for column in hue_cols
        ]

      # Add the is_categorical button
      colorscale_menu = [
        dict(
          label=str(color),
          method='restyle',
          args=[{'marker.colorscale': [color], 'marker.showscale':True}] if color is not None else [{'marker.showscale':False}],
          )
        for color in [None, "Jet", "Bluered", "YlOrRd", "Electric"] # Inderno, thermal, Agsunset dont work!!!
      ]
        
      plotly_fig.update_layout(
        updatemenus=[
          dict(
            buttons=hue_menu,
            direction="down",
            showactive=True,
            x=0.15,
            xanchor="left",
            y=1.1,
            yanchor="top"
            ),
          dict(
            buttons=colorscale_menu,
            direction="down",
            showactive=True,
            x=0.15,
            xanchor="left",
            y=0.95,
            yanchor="top"
            ),
          ],
        annotations=[
          dict(text="hue", x=0, xref="paper", y=1.07, yref="paper",
                              align="left", showarrow=False),
          dict(text="color", x=0, xref="paper", y=0.92,
                              yref="paper", showarrow=False),
          ]
        )
    
    return plotly_fig

  def plot_bokeh_fig(self, x, y, hue, hue_is_categorical, hue_name, marker_size, title='', hover_data=[]):
    '''
    x, y: strings, x, y keys in self.source
    hover_data: list of strings, keys in self.source to show in hover
    '''
    #! new
    TOOLTIPS = [(f"{x}", f"@{x}") for x in hover_data]
    hover = HoverTool(tooltips=TOOLTIPS)

    # набор инструментов
    #! new hover
    bokeh_fig = figure(title=title, tools=[hover, 'pan', 'wheel_zoom', 'box_select', 'lasso_select', 'reset', 'box_zoom'])

    if hue_is_categorical is None: # если не во что красить
      bokeh_fig.scatter(x=x, y=y, size=marker_size, source=self.source)

    elif hue_is_categorical: # Если hue категориальный, у нас будет легенда с возможностью спрятать отдельные hue
      # scatter -> label_name требует строку. Поэтому делаем из числовых категорий строки
      # Сортируем числа, потом делаем строки для корректной сортировки
      uniques = np.sort(hue.unique()).astype(str)

      # Настраиваем палитры
      n_unique = uniques.shape[0]
      if n_unique == 2:
        palette = tol['Bright'][3][:2]
      elif n_unique == 3:
        palette = tol['HighContrast'][3]
      elif n_unique in tol['Bright']:
        palette = tol['Bright'][n_unique]
      else:
        palette = magma(n_unique)

      # Делаем через for чтобы поддерживать legend.click_policy = 'hide'
      for i, hue_val in enumerate(uniques):
        # Будем рисовать только ту дату, где hue_col == hue_val
        condition = (hue.astype(str) == hue_val).tolist()
        view = CDSView(filter=BooleanFilter(condition))

        # Рисуем эмбеддинги
        bokeh_fig.scatter(x=x, y=y, size=marker_size,
                          source=self.source, view=view, legend_label=hue_val, color=palette[i])

      # Добавляем легенде возможность спрятать по клику
      bokeh_fig.legend.click_policy = 'hide'

    else: # Если hue числовой, у нас будет colorbar
      # Настраиваем цветовую палитру
      min_val, max_val = hue.min(), hue.max()
      color = linear_cmap(
        field_name=hue_name,
        palette=Inferno[256],
        low=min_val,
        high=max_val
      )

      # Рисуем эмбеддинги
      plot = bokeh_fig.scatter(x=x, y=y, size=marker_size, source=self.source, color=color)

      # Чуть настроим colorbar
      ticks = np.linspace(min_val, max_val, 5).round()
      ticker = FixedTicker(ticks=ticks)
      colorbar = plot.construct_color_bar(title=hue_name, title_text_font_size='20px', title_text_align='center',
                                          ticker=ticker, major_label_text_font_size='15px')
      bokeh_fig.add_layout(colorbar, 'below')

    bokeh_fig.title.align = 'center'
    
    return bokeh_fig
  
  def plot_embedding(self, embedding, hue=None, hue_is_categorical=False, hue_name=None, width=1200, height=500,
                      plotly_marker_size=1.5, bokeh_marker_size=3, title='', 
                      hover_data=None, cat_trsh=24, hue_cols=None):
    '''
    embedding: 2D/3D array
    hue: data[hue_col] в нужном формате (строки, если категории)
    hover_data: pd.DataFrame с данными для hover
    cat_trsh: int - порог для категориальных признаков
    hue_cols: pd.DataFrame с данными для dropdown menu динамической смены hue
    
    returns fig
    '''
    if embedding.shape[1] == 3:
      fig = self.plot_plotly_fig(embedding, hue, hue_name, title=title, 
                                  hover_data=hover_data, cat_trsh=cat_trsh, hue_cols=hue_cols)
      
      layout = fig.layout
      layout.update({'width': width, 'height': height,
                      'title_x': 0.5, 'title_font_size': 13, 'legend_itemsizing': 'constant'})
      new_fig = go.FigureWidget(fig.data, layout=layout)
      new_fig.update_traces(marker_size=plotly_marker_size)
      return new_fig
    else:
      if self.bokeh_first_time: # просто делаем CDS с embedding, hue
        self.source = ColumnDataSource({})
        self.bokeh_first_time = False
        
      x_name = f'{title}_x'
      y_name = f'{title}_y'
      x, y = embedding[:, 0], embedding[:, 1]
      self.source.data[x_name] = x
      self.source.data[y_name] = y
      if hover_data is not None:
        self.source.data.update(hover_data.to_dict('list'))
      hover_data = hover_data.columns if hover_data is not None else []
      if hue is not None:
        self.source.data[hue_name] = hue
          
      fig = self.plot_bokeh_fig(x_name, y_name, hue, hue_is_categorical, hue_name, marker_size=bokeh_marker_size, title=title, hover_data=hover_data)
      return fig
        
  def _plot_results(self, data, mapper_dict, default_hue_info, row_width, row_height, plotly_marker_size, bokeh_marker_size):
    '''
    по self.results и hue-информации рисует графики
    '''
    if default_hue_info is None:
      default_hue_info = None, None
    
    if self.bokeh_first_time:
      self.source = ColumnDataSource(data)
      self.bokeh_first_time = False
      output_notebook()
    
    plotly_figs, bokeh_figs = [], []
    
    # СБОР ФИГУР
    for mapper_name in mapper_dict:
      mapper_props = mapper_dict[mapper_name]
      hue_info = mapper_props.get('hue', default_hue_info)
      hue_field_name, hue_is_categorical = hue_info if hue_info is not None else (None, None)
      if hue_field_name is None:
        hue = None
      elif hue_is_categorical:
        # простой способ показывать легенду вместо colorbar
        hue = data[hue_field_name].astype(str)
      else:
        # в этом случае будет показываться colorbar
        hue = data[hue_field_name]
      
      embedding = self.results[mapper_name]['embedding']
      is_plotly = embedding.shape[1] == 3

      #! new
      if 'hover_data' in mapper_props and mapper_props['hover_data'] is not None:
        hover_data = data[mapper_props['hover_data']]
      else: hover_data = None

      #! new
      if 'hue_cols' in mapper_props and mapper_props['hue_cols'] is not None:
        hue_cols = data[mapper_props['hue_cols']]
      else: hue_cols = None

      #! new
      cat_trsh = mapper_props.get('cat_trsh', 24)

      fig = self.plot_embedding(embedding, hue, hue_is_categorical, hue_field_name,
                                300, row_height, plotly_marker_size, bokeh_marker_size, title=mapper_name, 
                                hover_data=hover_data, cat_trsh=cat_trsh, hue_cols=hue_cols)
      if is_plotly:
        plotly_figs.append(fig)
      else:
        bokeh_figs.append(fig)                
            
    # ОТРИСОВКА
    n_bokeh = len(bokeh_figs)
    if n_bokeh > 0:

      #! new / max(min(n_bokeh, 3), 1)
      plot_width = round(row_width / max(min(n_bokeh, 3), 1))

      #! new
      bokeh_figs = np.array(bokeh_figs +[None] * (-len(bokeh_figs)%3)).reshape(-1, 3).tolist()
      fig_grid = gridplot(bokeh_figs, width=plot_width, height=row_height)
      show(fig_grid)

    n_plotly = len(plotly_figs)
    if n_plotly > 0:

      #! new everything below
      plot_width = round(row_width / max(min(n_plotly, 3), 1))

      # сначала пихает в три колонки VBox, потом пихает все в строчку HBox
      column1_widgets = []
      column2_widgets = []
      column3_widgets = []
      for i in range(n_plotly):
        fig = plotly_figs[i]
        layout = fig.layout
        layout.update({'width': plot_width})
        new_fig = go.FigureWidget(fig.data, layout=layout)
        new_fig.update_traces(marker_size=plotly_marker_size)
        if i % 3 == 0: column1_widgets.append(new_fig)
        elif i % 3 == 1: column2_widgets.append(new_fig)
        else: column3_widgets.append(new_fig)

      column1 = widgets.VBox(column1_widgets)
      column2 = widgets.VBox(column2_widgets)
      column3 = widgets.VBox(column3_widgets)
      grid_layout = widgets.HBox([column1, column2, column3])
      display(grid_layout)            
          
  def plot_dim_reduction(self, data, mapper_dict, default_features=None, default_hue_info=None, reuse_results=True,
                      row_width=1500, row_height=500, plotly_marker_size=1.5, bokeh_marker_size=3, return_results=False):
    '''
    Метод принимает на вход данные и набор 2D/3D dimension-редукторов через mapper_dict.
    Отрисовывает эмбеддинги этих данных в наиболее удобных форматах: 3D - plotly, 2D - bokeh с CDS sharing'ом


    data - pd.DataFrame со всеми необходимыми данными - hue_cols, features

    mapper_dict - словарь знакомого вида :)

    default_features: array of strings - фичи которые будут использоваться для вычисления функции расстояния,
        если для reductor`а не указано иного

    default_hue_info: namedtuple - вида (hue-колонка-строка, is_categorical),
        инфа о hue-колонке, которая будет использоваться, если для reductor`а не указано иного

    row_width: int - ширина ряда из картинок
        узнать - рисуйте пустую bokeh.plotting.figure, увеличивая width,
        пока фигура не станет занимать все свободное место в ширину

    row_height: int
        желаемая высота ряда

    .._marker_size: размер точек на plotly и bokeh графиках

    return_results: bool - возвращать ли словарь {mapper_name: {'mapper': mapper, 'embedding': embedding}, ...}

    returns
        results: dict if return_results

        Note: для t-SNE mapper=embedding и лежит по ключу 'embedding'!
            Это объект класса TSNEEmbedding, это "обертка" над эмбеддингом.
            У него есть метод transform, а также его можно воспринимать как эмбеддинг и, например, слайсить и рисовать
    '''
    if not reuse_results or not hasattr(self, 'results'):
      results = self.get_results(data, mapper_dict, default_features)
      self.results = results
        
    self._plot_results(data, mapper_dict, default_hue_info, row_width, row_height, plotly_marker_size, bokeh_marker_size)
    
    if return_results:
      return self.results
