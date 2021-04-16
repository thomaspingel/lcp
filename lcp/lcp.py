import igraph
import numpy as np
import pandas as pd
import geopandas
from shapely.geometry import LineString
from skimage.graph import MCP_Geometric, MCP
from skimage import graph
from pyproj import Transformer

#%% LCP FUNCTIONS

# From Tobler. 1993. THREE PRESENTATIONS ON GEOGRAPHICAL ANALYSIS AND MODELING
# Returns velocity in km/hr
def cost_tobler_hiking_function(S,symmetric=True):

    # Convert to dz/dx
    S = np.tan(np.deg2rad(S))
    
    V = 6 * np.exp(-3.5 * np.abs(S + .05))
    
    if symmetric:
        V2 = 6 * np.exp(-3.5 * np.abs(-S + .05))
        V = (V + V2) / 2
        
    return 1 / V

#%%%
# From Rademaker et al. (2012)

# weight of traveler is given in kg
# weight of pack is given in kg
# terrain coefficients greater than 1 introduce "friction"
# velocity is Walking speed in meters per second

def cost_rademaker(S,weight=50,pack_weight=0,terrain_coefficient=1.1,velocity=1.2):
   
    # Rademaker assumes a grade in percent (0 to 100, rather than 0 to 1):
    G = 100 * np.arctan(np.deg2rad(S))
    
    W = weight
    L = pack_weight
    tc = terrain_coefficient
    V = velocity
    
    # Cost, in MWatts
    MW = 1.5*W + 2.0 * (W + L) * ((L/W)**2) + tc * (W+L) * (1.5 * V**2 + .35 * V * G)
    
    return MW


#%%

def cost_pingel_exponential(S,scale_factor=9.25):

    EXP = stats.expon.pdf(0,0,scale_factor) / stats.expon.pdf(S,0,scale_factor) 
    
    return EXP
    
#%%    
    
def ve(S,ve=2.3):
    S = np.tan(np.deg2rad(S))
    S = np.rad2deg(np.arctan(ve *  S))
    return S


#%%

def get_lists(nodes,edges):
    
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


#%%
def direct_routes(nodes,edges):
    
    start_list, end_list, ids, start_coords, end_coords = get_lists(nodes,edges)

    gdf = pd.DataFrame()
    
    for i,this_start in enumerate(start_coords):
        df = pd.DataFrame()
        these_end_coords = end_coords[i]
        df['ids'] = ids[i]
        df['method'] = 'direct'
        df['geometry'] = [LineString([this_start,this_end]) for this_end in these_end_coords]
        
        gdf = gdf.append(df,ignore_index=True)

    gdf = geopandas.GeoDataFrame(gdf,geometry=gdf['geometry'],crs=4326)
    
    return gdf

#%%

def lcp_coordinate_conversion(start_coords,end_coords,crs,transform):
    
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
   



#%%

def get_areal_routes(nodes,edges,surface,meta,label='areal'):
    
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
        df['method'] = label
        df['geometry'] = geometries
        gdf = gdf.append(df,ignore_index=True)
        
        gdf = geopandas.GeoDataFrame(gdf,geometry=gdf['geometry'],crs=meta['crs'])
        
    
    return gdf


#%% 

def create_raster_network(X):
    
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


#%%

def get_linear_routes(nodes,edges,df,meta,label='linear'):
    
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
        df['method'] = label
        df['geometry'] = geometries
        gdf = gdf.append(df,ignore_index=True)

    gdf = geopandas.GeoDataFrame(gdf,geometry=gdf['geometry'],crs=meta['crs'])
    
    return gdf

def coord_transform(x,y,from_epsg,to_epsg):
    transformer = Transformer.from_crs(from_epsg,to_epsg,always_xy=True)
    return transformer.transform(x,y)
