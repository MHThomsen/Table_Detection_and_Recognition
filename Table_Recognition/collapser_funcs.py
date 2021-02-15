
#En række funktioner som tager slice som input, og returnerer et 1d array med de ønskede features


class collapser_function():
    def __init__(self):
        self.out_dim = None

    def is_correct_size(slice3d):
        #Expects input to be of shape CxWxH, but if not, 
        assert len(slice3d.shape) == 3, f"Slice er i {len(slice3d.shape)} dimensioner. Vi burde kun have et CxWxH input (ingen batch dimension)"    
        


class min_collapser(collapser_function):

    def __init__(self,
                 slice_channels,
                 slice_width,
                 slice_height):
        self.out_dim = slice_width*slice_height
    
    def collapse(slice3d, *args):
        super.is_correct_size(slice3d)
        return torch.min(slice3d,dim=0)[0].reshape(1,-1)
        
    
class max_collapser(collapser_function):

    def __init__(self,
                 slice_channels,
                 slice_width,
                 slice_height):
        self.out_dim = slice_width*slice_height
    
    def collapse(slice3d, *args):
        super.is_correct_size(slice3d)
        return torch.max(slice3d,dim=0)[0].reshape(1,-1)    

class median_collapser(collapser_function):

    def __init__(self,
                 slice_channels,
                 slice_width,
                 slice_height):
        self.out_dim = slice_width*slice_height
    
    def collapse(slice3d, *args):
        super.is_correct_size(slice3d)
        return torch.median(slice3d,dim=0)[0].reshape(1,-1)    

class mean_channel_collapser(collapser_function):

    def __init__(self,
                 slice_channels,
                 slice_width,
                 slice_height):
        
         """
        Calculate mean of all WxH pixels, across all channels. 
        Most intuitive; get mean of 2d slice
        """
        
        self.out_dim = slice_width*slice_height
    
    def collapse(slice3d, *args):
        super.is_correct_size(slice3d)
        
        #OBS: intet [0] index på denne inden reshape. 
        return torch.mean(slice3d,dim=0).reshape(1,-1)   


class mean_2d_collapser(collapser_function):

    def __init__(self,
                 slice_channels,
                 slice_width,
                 slice_height):
        
        """
        Calculate mean for each of the 2d slices, returning the mean value of each of the channels.
        Not as intuitive as mean_channel_collapser, however might make a lot of sence to create a feature 
        of average activation across each channel. Might not at all
        """
        
        self.out_dim = slice_channels
    
    def collapse(slice3d,*args):
        super.is_correct_size(slice3d)
        
        #No need to reshape 
        return torch.mean(torch.mean(slice3d,dim=1),dim=1)  

    
class slice_collapser(collapser_function):

    def __init__(self,
                 slice_channels,
                 slice_width,
                 slice_height):
        
        self.out_dim = slice_width*slice_height
    
    def collapse(slice3d,
                 channel_no = 0):
        
        super.is_correct_size(slice3d)
        
        """
        Takes channel_no as input, determining which of the channels to extract
        """
       
        return slice3d[channel_no,:,:].reshape(1,-1)





    
    
    
#Fordelen ved at lave *args i de forskellige funktioner er at vi kan definere en generel funktion
#self.collapser_func = min_collapser(),
#og så kan vi kalde self.collapser_func(slice,0), selvom kun "slice_collapser" faktisk bruger 2. argument 





