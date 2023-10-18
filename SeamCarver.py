import cv2
import numpy as np

class SeamCarver:
    def __init__(self, filename, out_height, out_width):
        # intialization
        self.filename=filename
        self.out_height=out_height
        self.out_width=out_width
        
        #read image
        self.in_image=cv2.imread(filename).astype(np.float64)
        self.in_height,self.in_width=self.in_image.shape[:2]
        self.out_image=np.copy(self.in_image)
        
        # kernel for dp
        self.kernel_x = np.array([[0., 0., 0.], [-1., 0., 1.], [0., 0., 0.]], dtype=np.float64)
        self.kernel_y_left = np.array([[0., 0., 0.], [0., 0., 1.], [0., -1., 0.]], dtype=np.float64)
        self.kernel_y_right = np.array([[0., 0., 0.], [1., 0., 0.], [0., -1., 0.]], dtype=np.float64)
        
    def seam_carving(self):
        delta_width=int(self.out_width-self.in_width)
        if delta_width<0:
            self.seam_remove(delta_width*-1)
            
    def seam_remove(self,delta_width):
        for i in range(delta_width):
            energy_map = self.calc_energy_map()
            dp_result=self.dynamic_program(energy_map)
            shortest_seam=self.find_seam(dp_result)
            self.delete_seam(shortest_seam)
            
    def calc_energy_map(self):
        #calc gradient
        b,g,r=cv2.split(self.out_image)
        b_energy = np.absolute(cv2.Scharr(b, -1, 1, 0)) + np.absolute(cv2.Scharr(b, -1, 0, 1))
        g_energy = np.absolute(cv2.Scharr(g, -1, 1, 0)) + np.absolute(cv2.Scharr(g, -1, 0, 1))
        r_energy = np.absolute(cv2.Scharr(r, -1, 1, 0)) + np.absolute(cv2.Scharr(r, -1, 0, 1))
        return b_energy+g_energy+r_energy
    
    def dynamic_program(self,energy_map):
        matrix_x = self.neighbor_matrix(self.kernel_x)
        matrix_y_left = self.neighbor_matrix(self.kernel_y_left)
        matrix_y_right = self.neighbor_matrix(self.kernel_y_right)
        #dp
        row,col=energy_map.shape
        ret=np.copy(energy_map)
        for r in range(row):
            for c in range(col):
                if c==0:
                    energy_right=ret[r-1,c+1]+matrix_x[r-1,c+1]+matrix_y_right[r-1,c+1]
                    energy_up=ret[r-1,c]+matrix_x[r-1,c]
                    ret[r,c]=energy_map[r,c]+min(energy_right,energy_up)
                elif c==col-1:
                    energy_left=ret[r-1,c-1]+matrix_x[r-1,c-1]+matrix_y_left[r-1,c-1]
                    energy_up=ret[r-1,c]+matrix_x[r-1,c]
                    ret[r,c]=energy_map[r,c]+min(energy_left,energy_up)
                else:
                    energy_left=ret[r-1,c-1]+matrix_x[r-1,c-1]+matrix_y_left[r-1,c-1]
                    energy_up=ret[r-1,c]+matrix_x[r-1,c]
                    energy_right=ret[r-1,c+1]+matrix_x[r-1,c+1]+matrix_y_right[r-1,c+1]
                    ret[r,c]=energy_map[r,c]+min(energy_left,energy_up,energy_right)
                    
        return ret
    
        
    def neighbor_matrix(self,kernel):
        b,g,r=cv2.split(self.in_image)
        #convolution
        ret=np.absolute(cv2.filter2D(b,-1,kernel=kernel))+np.absolute(cv2.filter2D(g,-1,kernel=kernel))+np.absolute(cv2.filter2D(r, -1, kernel=kernel))
        return ret
    
    def find_seam(self,calc_map):
        row,col=calc_map.shape
        ret=np.zeros((row,),dtype=np.uint32)
        ret[-1]=np.argmin(calc_map[-1])
        for r in range(row-2,-1,-1):
            prev_col=ret[r+1]
            if prev_col==0:
                ret[r]=np.argmin(calc_map[r,:2])
            elif prev_col==col-1:
                ret[r]=np.argmin(calc_map[r,col-2:col]) + prev_col-1
            else:
                ret[r]=np.argmin(calc_map[r,prev_col-1:prev_col+2])+prev_col-1
        return ret
    
    def delete_seam(self,found_seam):
        row,col=self.out_image.shape[:2]
        result=np.zeros((row,col-1,3))
        for r in range(row):
            c=found_seam[r]
            result[r,:,0]=np.delete(self.out_image[r,:,0],[c])
            result[r,:,1]=np.delete(self.out_image[r,:,1],[c])
            result[r,:,2]=np.delete(self.out_image[r,:,2],[c])
        self.out_image=np.copy(result)
    
    def save_result(self,filename):
        cv2.imwrite(filename, self.out_image.astype(np.uint8))
        
        