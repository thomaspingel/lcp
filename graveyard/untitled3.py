# -*- coding: utf-8 -*-
"""
Created on Fri Apr 16 22:58:28 2021

@author: Thomas Pingel
"""

import numpy as np
import igraph
import lcp
from joblib import Parallel, delayed

import time
from numba import jit

#%%
tic = time.time()
k = 1000
X = np.random.rand(k,k)
df = lcp.create_raster_network(X)
df['weight'] = df.loc[:,['source_value','target_value']].mean(axis=1)

G = igraph.Graph()
G.add_vertices(k**2)
G.add_edges(list(zip(df.source,df.target)),attributes={'weight':df.weight})

starts = list(np.random.randint(0,k**2-1,10))
ends = [list(np.random.randint(0,k**2-1,10)) for i in range(len(starts))]
toc = time.time()
print(toc-tic)

#%%
tic = time.time()
routes = [G.get_shortest_paths(item[0],item[1],weights='weight') for item in zip(starts,ends)]
toc = time.time()
print(toc-tic)

#%%
tic = time.time()
#params = list(zip(starts,ends))
result = Parallel(n_jobs=-1)(delayed(G.copy().get_shortest_paths)(item[0],item[1],weights='weight') for item in zip(starts,ends))
toc = time.time()
print(toc-tic)
#%%

def cost_tobler_hiking_function(S,symmetric=True):
    """
    Applies Tobler's Hiking Function to slope data supplied in DEGREES.

    From Tobler. 1993. Three Presentation on Geographical Analysis and Modeling.
    
    Simple Example:

    C = lcp.cost_tobler_hiking_function(S,symmetric=True)
    
    Parameters:
    - 'S' is an array (any dimension) of slope values in DEGREES.
    - 'symmetric' flags whether to consider slope values symmetrically.  Note that this_end
            is NOT the same as just taking the positive values.  This returns an average
            of the positive and negative value for the given slope.
    
    Returns:
    - 'C' a cost surface of velocity in km/hr

    """    
    
    # Convert to dz/dx
    S = np.tan(np.deg2rad(S))
    
    V = 6 * np.exp(-3.5 * np.abs(S + .05))
    
    if symmetric:
        V2 = 6 * np.exp(-3.5 * np.abs(-S + .05))
        V = (V + V2) / 2
        
    return 1 / V


def cost_rademaker(S,weight=50,pack_weight=0,terrain_coefficient=1.1,velocity=1.2):
    """
    Applies Rademaker et al's model (2012) to slope values for LCP calculation.
    
    Simple Example:

    C = lcp.cost_rademaker(S,weight=50,pack_weight=0,terrain_coefficient=1.1,velocity=1.2)
    
    Parameters:
    - 'S' is an array (any dimension) of slope values in DEGREES.
    - 'weight' is weight of traveler is given in kg
    - 'pack_weight' is cargo weight, given in kg
    - 'terrain_coefficient' is a value to introduce "friction".  Values greater than
            one have more than 'average' friction.
    - 'velocity' is mean walking speed in meters per second
    
    Returns:
    - 'C' a cost surface of shape S.

    """     
    # Rademaker assumes a grade in percent (0 to 100, rather than 0 to 1):
    G = 100 * np.arctan(np.deg2rad(S))
    
    W = weight
    L = pack_weight
    tc = terrain_coefficient
    V = velocity
    
    # Cost, in MWatts
    MW = 1.5*W + 2.0 * (W + L) * ((L/W)**2) + tc * (W+L) * (1.5 * V**2 + .35 * V * G)
    
    return MW



def cost_pingel_exponential(S,scale_factor=9.25):
    """
    Applies the exponental LCP cost function described by Pingel (2010). 
    
    Simple Example:

    C = lcp.cost_pingel_exponential(S,scale_factor=9.25)
    
    Parameters:
    - 'S' is an array (any dimension) of slope values in DEGREES.
    - 'scale_factor' is a value in degrees that generally corresponds to the mean slope
      (in degrees) of a path network.  Larger values represent a larger tolerance for 
      steeper slopes.  Smaller values will cause an LCP to avoid steeper slopes.

    """  
    
    EXP = stats.expon.pdf(0,0,scale_factor) / stats.expon.pdf(S,0,scale_factor) 
    
    return EXP
    
 
    
def ve(S,ve=2.3):
    """
    Applies a vertical exaggeration to a slope raster and returns it.  Slope raster must be in DEGREES.
    
    Simple Example:

    S_ve = lcp.ve(S,2.3)

    """      
    S = np.tan(np.deg2rad(S))
    S = np.rad2deg(np.arctan(ve *  S))
    return S



def get_lists(nodes,edges):
    """
    Simple Example:

    start_list, end_list, ids, start_coords, end_coords = lcp.get_lists(nodes, edges)    
    
    Internal method to transform nodes and edges into lists of start coords and lists of lists of end coords.
    
    Returns: start_list, end_list, ids, start_coords, end_coords
    
    """        
    nodes['coords'] = list(zip(nodes.iloc[:,0], nodes.iloc[:,1]))  
    
    start_list = edges.iloc[:,0].unique()
    end_list = [edges.iloc[:,1].loc[edges.iloc[:,0]==item].values for item in start_list] 
    
    start_coords = []
    end_coords = []
    ids = []
    
    for i, this_start in enumerate(start_list):
        these_ends = end_list[i]
        
        these_ids = [this_start + '_to_' + te for te in these_ends]
        these_start_coords = nodes.loc[this_start,'coords']
        these_end_coords = nodes.loc[these_ends,'coords'].values
        
        start_coords.append(these_start_coords)
        end_coords.append(these_end_coords)
        ids.append(these_ids)


    return start_list, end_list, ids, start_coords, end_coords


def direct_routes(nodes,edges):
    """
    
    Returns a straight-line path between edges.
    
    Simple Example:

    gdf = lcp.direct_routes(nodes, edges)
    
    Parameters:
    - 'nodes' is a Pandas DataFrame where the first column is a unique ID, the second is
                an x coordinate (e.g., longitude) and the third is a y coordinate (e.g.,
                latitude).
    - 'edges' is a Pandas DataFrame were the first column is a source ID (matching a node)
                and the second column is a destination.  At the moment, we assume no 
                directionality / edges are symmetric.
    - 'array' is a numpy array representing the cost surface.
    - 'meta' is a dictionary, that must contain 'crs' and 'transform' items corresponding
                to those returned by rasterio.  neilpy.imread returns such a dictionary
                by default.
    - 'label' is used to identify the type of cost path/surface in the GeoDataFrame output
                rows.
                
    Output:
    - 'gdf' is a GeoPandas GeoDataFrame with fields 'ids' describing the source and target 
                , 'label' corresponding to the label, and a geometry field containing the
                path in shapely / WKT format.                

    """      
    start_list, end_list, ids, start_coords, end_coords = get_lists(nodes,edges)

    gdf = pd.DataFrame()
    
    for i,this_start in enumerate(start_coords):
        df = pd.DataFrame()
        these_end_coords = end_coords[i]
        df['ids'] = ids[i]
        df['label'] = 'direct'
        df['geometry'] = [LineString([this_start,this_end]) for this_end in these_end_coords]
        
        gdf = gdf.append(df,ignore_index=True)

    gdf = geopandas.GeoDataFrame(gdf,geometry=gdf['geometry'],crs=4326)
    
    return gdf



def lcp_coordinate_conversion(start_coords,end_coords,crs,transform):
    """
    Simple Example:

    network = lcp.create_raster_network(array)
    
    Parameters:
    - 'start_coords' is a list of tuples (lon,lat)
    - 'end_coords' is a list of lists of tuples.  Each list of end points corresponds to 
           a start point, so len(start_coords) must equal len(end_coords), although each 
           list OF end points can be of any length one or greater.
    - 'crs' is a Coordinate Reference System of the type returned by rasterio (or neilpy).
    - 'transform' is an Affine transformation matrix as returned by rasterio (or neilpy).
                
    Output:
    - 'converted_start_coords' is a list of tuples of PIXEL coordinates.
    - 'converted_end_coords' is a list of list of tupes of pixel coordiantes.

    """       
    converted_start_coords = []
    converted_end_coords = []
    
    for i,this_start_coord in enumerate(start_coords):
        these_end_coords = end_coords[i]
        
        # Convert from lat/lon to map coordinates
        this_start_coord = coord_transform(*this_start_coord,4326,crs)
        these_end_coords = [coord_transform(*item,4326,crs) for item in these_end_coords]
        
        # Convert from map coordinates to pixel coordinates
        this_start_coord = (~transform*this_start_coord)[::-1]
        these_end_coords = [(~transform*item)[::-1] for item in these_end_coords]
        
        # Round them to ints
        this_start_coord = tuple(np.round(this_start_coord).astype(np.uint32))
        these_end_coords = [tuple(item) for item in np.round(these_end_coords).astype(np.uint32)]
        
        converted_start_coords.append(this_start_coord)
        converted_end_coords.append(these_end_coords)
        
    return converted_start_coords, converted_end_coords
   

def get_areal_routes(nodes,edges,surface,meta,label='areal'):
    """
    Simple Example:

    gdf = lcp.get_areal_routes(nodes, edges, array, meta, label)
    
    Parameters:
    - 'nodes' is a Pandas DataFrame where the first column is a unique ID, the second is
                an x coordinate (e.g., longitude) and the third is a y coordinate (e.g.,
                latitude).
    - 'edges' is a Pandas DataFrame were the first column is a source ID (matching a node)
                and the second column is a destination.  At the moment, we assume no 
                directionality / edges are symmetric.
    - 'array' is a numpy array representing the cost surface.
    - 'meta' is a dictionary, that must contain 'crs' and 'transform' items corresponding
                to those returned by rasterio.  neilpy.imread returns such a dictionary
                by default.
    - 'label' is used to identify the type of cost path/surface in the GeoDataFrame output
                rows.
                
    Output:
    - 'gdf' is a GeoPandas GeoDataFrame with fields 'ids' describing the source and target 
                , 'label' corresponding to the label, and a geometry field containing the
                path in shapely / WKT format.                

    """        
    gdf = pd.DataFrame()

    print('Creating surface network for',label)
    m = MCP_Geometric(surface,fully_connected=True)
    print('Done creating surface network.')
    
    start_list, end_list, ids, start_coords, end_coords = get_lists(nodes,edges)
    
    conv_start_coords, conv_end_coords = lcp_coordinate_conversion(start_coords,end_coords,meta['crs'],meta['transform'])
    
    for i,this_start_coord in enumerate(conv_start_coords):
        these_end_coords = conv_end_coords[i]

        print('Calculating costs and routes.')
        costs, traceback_array  = m.find_costs([this_start_coord],these_end_coords,find_all_ends=True)
        print('Done calculating costs and routes.')
        
        # Pull routes and convert
        routes = [m.traceback(this_end_coord) for this_end_coord in these_end_coords]
        geometries= [LineString(np.vstack(meta['transform']*np.fliplr(route).T).T) for route in routes]
        
        df = pd.DataFrame()
        df['ids'] = ids[i]
        df['label'] = label
        df['geometry'] = geometries
        gdf = gdf.append(df,ignore_index=True)
        
        gdf = geopandas.GeoDataFrame(gdf,geometry=gdf['geometry'],crs=meta['crs'])
        
    
    return gdf


@jit(nopython=True)
def create_raster_network(X):
    """
    Simple Example:

    network = lcp.create_raster_network(array)
    
    Parameters:
    - 'array' is a numpy array representing the cost surface.
                
    Output:
    - 'network' is a Pandas DataFrame with fields 'source' and 'target' representing 1D
           (flattened) indices, source_value and target_value for pixel data, 'distance'
           which is the pixel distance (1 for orthogonal, 2**.5 for diagonal).  These
           should be used directly by the operator to calculate a 'weight' field 
           before passing to lcp.get_linear_routes()           

    """       
    m,n = np.shape(X)
    
    I = np.reshape(np.arange(np.size(X),dtype=np.int32),np.shape(X))

    df = pd.DataFrame()
    df['source'] = np.hstack((I[1:,1:].flatten(),
                              I[1:,:].flatten(),
                              I[1:,:-1].flatten(),
                              I[:,:-1].flatten()))
    df['target'] = np.hstack((ashift(I,0)[1:,1:].flatten(),
                              ashift(I,1)[1:,:].flatten(),
                              ashift(I,2)[1:,:-1].flatten(),
                              ashift(I,3)[:,:-1].flatten()))

    df['source_value'] = X.flatten()[df['source'].values]
    df['target_value'] = X.flatten()[df['target'].values]

    df['distance'] = np.hstack((2**.5*np.ones((m-1)*(n-1)),
                                np.ones(n*(m-1)),
                                2**.5*np.ones((m-1)*(n-1)),
                                np.ones(m*(n-1))))
    
    return df




def get_linear_routes(nodes,edges,df,meta,label='linear'):
    """
    Simple Example:

    network = lcp.create_raster_network(array)
    network['weight'] = np.abs(network['source_value'] - network['target_value']) / network['distance']
    gdf = lcp.get_linear_routes(nodes, edges, network, meta, label)
    
    Parameters:
    - 'nodes' is a Pandas DataFrame where the first column is a unique ID, the second is
                an x coordinate (e.g., longitude) and the third is a y coordinate (e.g.,
                latitude).
    - 'edges' is a Pandas DataFrame were the first column is a source ID (matching a node)
                and the second column is a destination.  At the moment, we assume no 
                directionality / edges are symmetric.
    - 'network' is a Pandas DataFrame created by lcp.create_raster_network().  It MUST 
                include a column called 'weight'.
    - 'meta' is a dictionary, that must contain 'crs' and 'transform' items corresponding
                to those returned by rasterio.  It must also contain 'height' and
                'width' items.  neilpy.imread returns such a dictionary by default.
    - 'label' is used to identify the type of cost path/surface in the GeoDataFrame output
                rows.
                
    Output:
    - 'gdf' is a GeoPandas GeoDataFrame with fields 'ids' describing the source and target 
                , 'label' corresponding to the label, and a geometry field containing the
                path in shapely / WKT format.                

    """         
    
    img_dim = (meta['height'],meta['width'])

    G = igraph.Graph()
    G.add_vertices(img_dim[0] * img_dim[1])
    G.add_edges(list(zip(df.source,df.target)),attributes={'weight':df.weight})
    del df

    gdf = pd.DataFrame()
    start_list, end_list, ids, start_coords, end_coords = get_lists(nodes,edges)
    conv_start_coords, conv_end_coords = lcp_coordinate_conversion(start_coords,end_coords,meta['crs'],meta['transform']) 
    
    for i,this_start_coord in enumerate(conv_start_coords):
        these_end_coords = conv_end_coords[i]  
        flat_start = np.ravel_multi_index(this_start_coord,img_dim)
        flat_ends = [np.ravel_multi_index(item,img_dim) for item in these_end_coords]
        routes = G.get_shortest_paths(flat_start,flat_ends,weights='weight')
        routes2 = [np.flipud(np.vstack(np.unravel_index(route,img_dim))) for route in routes]
        geometries = [LineString(np.vstack(meta['transform']*route2).T) for route2 in routes2] 
        df = pd.DataFrame()
        df['ids'] = ids[i]
        df['label'] = label
        df['geometry'] = geometries
        gdf = gdf.append(df,ignore_index=True)

    gdf = geopandas.GeoDataFrame(gdf,geometry=gdf['geometry'],crs=meta['crs'])
    
    return gdf

def coord_transform(x,y,from_epsg,to_epsg):
    transformer = Transformer.from_crs(from_epsg,to_epsg,always_xy=True)
    return transformer.transform(x,y)
    
def ashift(surface,direction,n=1):
    surface = surface.copy()
    if direction==0:
        surface[n:,n:] = surface[0:-n,0:-n]
    elif direction==1:
        surface[n:,:] = surface[0:-n,:]
    elif direction==2:
        surface[n:,0:-n] = surface[0:-n,n:]
    elif direction==3:
        surface[:,0:-n] = surface[:,n:]
    elif direction==4:
        surface[0:-n,0:-n] = surface[n:,n:]
    elif direction==5:
        surface[0:-n,:] = surface[n:,:]
    elif direction==6:
        surface[0:-n,n:] = surface[n:,0:-n]
    elif direction==7:
        surface[:,n:] = surface[:,0:-n]
    return surface
