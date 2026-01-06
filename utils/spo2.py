import numpy as np
# import scipy.signal

class HeartRateAndOxygenSaturation:
    def __init__(self, buffer_size=250, ma4_size=4, freqs=62.5):

        self.BUFFER_SIZE = buffer_size  
        self.MA4_SIZE = ma4_size  
        self.FreqS = freqs  
        self.uch_spo2_table = [ 95, 95, 95, 96, 96, 96, 97, 97, 97, 97, 97, 98, 98, 98, 98, 98, 99, 99, 99, 99, 
              99, 99, 99, 99, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 
              100, 100, 100, 100, 99, 99, 99, 99, 99, 99, 99, 99, 98, 98, 98, 98, 98, 98, 97, 97, 
              97, 97, 96, 96, 96, 96, 95, 95, 95, 94, 94, 94, 93, 93, 93, 92, 92, 92, 91, 91, 
              90, 90, 89, 89, 89, 88, 88, 87, 87, 86, 86, 85, 85, 84, 84, 83, 82, 82, 81, 81, 
              80, 80, 79, 78, 78, 77, 76, 76, 75, 74, 74, 73, 72, 72, 71, 70, 69, 69, 68, 67, 
              66, 66, 65, 64, 63, 62, 62, 61, 60, 59, 58, 57, 56, 56, 55, 54, 53, 52, 51, 50, 
              49, 48, 47, 46, 45, 44, 43, 42, 41, 40, 39, 38, 37, 36, 35, 34, 33, 31, 30, 29, 
              28, 27, 26, 25, 23, 22, 21, 20, 19, 17, 16, 15, 14, 12, 11, 10, 9, 7, 6, 5, 
              3, 2, 1 ]
        self.an_x = np.zeros(self.BUFFER_SIZE) # ir
        self.an_y = np.zeros(self.BUFFER_SIZE) # red
        self.spo2_ir = []
        self.spo2_red = []
        self.pn_spo2 = 0
        self.pch_spo2_valid = 0 
        self.pn_heart_rate = 0
        self.pch_hr_valid = 0

    def maxim_get_spo2_vs_heart_rate(self, ir_data, red_data):
        # Append data to the SPO2 array
        self.spo2_red.extend(red_data)
        self.spo2_ir.extend(ir_data)
        if len(self.spo2_ir) > 250:
            self.pn_spo2, self.pch_spo2_valid, self.pn_heart_rate, self.pch_hr_valid = self.maxim_heart_rate_and_oxygen_saturation(np.array(self.spo2_ir[:250]), np.array(self.spo2_red[:250]))
            # print(f"HR={self.pn_heart_rate}, HRvalid={self.pch_hr_valid}, SPO2={self.pn_spo2}, SPO2Valid={self.pch_spo2_valid}")

            # Remove relatively the first second of data, nearly 62 samples due to the sampling rate 62.5Hz
            self.spo2_ir = self.spo2_ir[62:]
            self.spo2_red = self.spo2_red[62:]

        return self.pn_spo2, self.pch_spo2_valid, self.pn_heart_rate, self.pch_hr_valid

    def maxim_heart_rate_and_oxygen_saturation(self, pun_ir_buffer, pun_red_buffer, low_threshold=30, high_threshold=60):
        data_len = len(pun_ir_buffer)
        # Initialize the all required variable
        an_ir_valley_locs = np.zeros(15)
        an_ratio = np.zeros(5)
        # n_ir_buffer_length = len(pun_ir_buffer)
        n_npks = 0

        #  calculates DC mean and subtract DC from ir
        un_ir_mean = np.mean(pun_ir_buffer)

        # remove DC and invert signal so that we can use peak detector as valley detector
        self.an_x[:] = -1*(pun_ir_buffer - un_ir_mean)
        
        # 4 pt Moving Average
        for k in range(self.BUFFER_SIZE - self.MA4_SIZE):
            self.an_x[k] = np.mean(self.an_x[k:k+4])    

        # calculate threshold  
        n_th1 = np.clip(np.mean(self.an_x), low_threshold, high_threshold)

        # since we flipped signal, we use peak detector as valley detector
        an_ir_valley_locs, n_npks = self.maxim_find_peaks(an_ir_valley_locs, n_npks, self.an_x, self.BUFFER_SIZE, n_th1, 20, 15) # peak_height, peak_distance, max_num_peaks 

        n_peak_interval_sum = np.diff(an_ir_valley_locs[:n_npks]).sum()

        if n_npks >= 2:
            pn_heart_rate = int((self.FreqS*60)/ (n_peak_interval_sum / (n_npks-1)))
            pch_hr_valid = 1
        else:
            pn_heart_rate = -999
            pch_hr_valid = 0

        # load raw value again for SPO2 calculation : RED(=y) and IR(=X)
        self.an_x[:] = pun_ir_buffer
        self.an_y[:] = pun_red_buffer

        # find precise min near an_ir_valley_locs
        n_exact_ir_valley_locs_count = n_npks

        # using exact_ir_valley_locs , find ir-red DC andir-red AC for SPO2 calibration an_ratio
        # finding AC/DC maximum of raw
        n_ratio_average = 0
        n_i_ratio_count = 0 

        for k in range(n_exact_ir_valley_locs_count):
                if an_ir_valley_locs[k] > self.BUFFER_SIZE:
                    pn_spo2 = -999 # do not use SPO2 since valley loc is out of range
                    pch_spo2_valid = 0
                    return pn_spo2, pch_spo2_valid, pn_heart_rate, pch_hr_valid

        # find max between two valley locations 
        # and use an_ratio betwen AC compoent of Ir & Red and DC compoent of Ir & Red for SPO2 
        for k in range(n_exact_ir_valley_locs_count-1):
            n_y_dc_max = -16777216
            n_x_dc_max = -16777216
            if an_ir_valley_locs[k+1] - an_ir_valley_locs[k] > 3:
                for i in range(an_ir_valley_locs[k], an_ir_valley_locs[k+1]):
                    if self.an_x[i] > n_x_dc_max:
                        n_x_dc_max = self.an_x[i]
                        n_x_dc_max_idx = i
                    if self.an_y[i] > n_y_dc_max:
                        n_y_dc_max = self.an_y[i]
                        n_y_dc_max_idx = i

                n_y_ac = (self.an_y[an_ir_valley_locs[k+1]] - self.an_y[an_ir_valley_locs[k]]) * (n_y_dc_max_idx - an_ir_valley_locs[k])
                n_y_ac = self.an_y[an_ir_valley_locs[k]] + n_y_ac / (an_ir_valley_locs[k+1] - an_ir_valley_locs[k])
                n_y_ac = self.an_y[n_y_dc_max_idx] - n_y_ac

                n_x_ac = (self.an_x[an_ir_valley_locs[k+1]] - self.an_x[an_ir_valley_locs[k]]) * (n_x_dc_max_idx - an_ir_valley_locs[k])
                n_x_ac = self.an_x[an_ir_valley_locs[k]] + n_x_ac / (an_ir_valley_locs[k+1] - an_ir_valley_locs[k])
                n_x_ac = self.an_x[n_x_dc_max_idx] - n_x_ac

                # n_nume = (n_y_ac * n_x_dc_max) >> 7
                n_nume = (n_y_ac * n_x_dc_max) / 128

                # n_denom = (n_x_ac * n_y_dc_max) >> 7
                n_denom = (n_x_ac * n_y_dc_max) / 128

                if n_denom > 0 and n_i_ratio_count < 5 and n_nume != 0:
                    an_ratio[n_i_ratio_count] = (n_nume * 100) / n_denom
                    n_i_ratio_count += 1

        # choose median value since PPG signal may varies from beat to beat
        an_ratio = self.maxim_sort_ascend(an_ratio, n_i_ratio_count)
        n_middle_idx = n_i_ratio_count // 2

        if n_middle_idx > 1:
            n_ratio_average = (an_ratio[n_middle_idx-1] + an_ratio[n_middle_idx]) // 2
        else:
            n_ratio_average = an_ratio[n_middle_idx]

        if 2 < n_ratio_average < 183:
            # you need to provide uch_spo2_table yourself
            n_spo2_calc = self.uch_spo2_table[int(n_ratio_average)]
            pn_spo2 = n_spo2_calc
            pch_spo2_valid = 1
        else:
            pn_spo2 = -999
            pch_spo2_valid = 0

        return pn_spo2, pch_spo2_valid, pn_heart_rate, pch_hr_valid


    def mean(self, data):
        return float(sum(data)) / max(len(data), 1)

    def sort_ascend(self, x):
        return sorted(x)

    def maxim_find_peaks(self, pn_locs, pn_npks, pn_x, n_size, n_min_height, n_min_distance, n_max_num):
        """
        Find at most MAX_NUM peaks above MIN_HEIGHT separated by at least MIN_DISTANCE
        """
        pn_locs, pn_npks = self.maxim_peaks_above_min_height(pn_locs, pn_npks, pn_x, n_size, n_min_height)
        pn_locs, pn_npks = self.maxim_remove_close_peaks(pn_locs, pn_npks, pn_x, n_min_distance)
        pn_npks = min(pn_npks, n_max_num)

        return pn_locs, pn_npks


    def maxim_peaks_above_min_height(self, pn_locs, n_npks, pn_x, n_size, n_min_height):
        """
        Find peaks above n_min_height
        """
        i = 1
        n_npks = 0
        pn_locs = []
        
        while i < n_size - 1:
            if pn_x[i] > n_min_height and pn_x[i] > pn_x[i-1]:      # find left edge of potential peaks
                n_width = 1
                while i+n_width < n_size and pn_x[i] == pn_x[i+n_width]:  # find flat peaks
                    n_width += 1

                if i + n_width >= n_size:
                    break   

                if pn_x[i] > pn_x[i+n_width] and n_npks < 15:      # find right edge of peaks
                    pn_locs.append(i)
                    n_npks += 1
                    # for flat peaks, peak location is left edge
                    i += n_width + 1
                else:
                    i += n_width

            else:
                i += 1
        return pn_locs, n_npks
    


    def maxim_remove_close_peaks(self, pn_locs, pn_npks, pn_x, n_min_distance):
        """
        Remove peaks separated by less than MIN_DISTANCE
        """
        # Order peaks from large to small
        pn_locs = self.maxim_sort_indices_descend(pn_x, pn_locs, pn_npks)

        i = -1
        while i < pn_npks:
            n_old_npks = pn_npks
            pn_npks = i + 1
            j = i + 1
            while j < n_old_npks:
                n_dist = pn_locs[j] - (-1 if i == -1 else pn_locs[i])  # lag-zero peak of autocorr is at index -1
                if n_dist > n_min_distance or n_dist < -n_min_distance:
                    pn_locs[pn_npks] = pn_locs[j]
                    pn_npks += 1
                j += 1
            i += 1

        # Resort indices into ascending order
        pn_locs[:pn_npks] = self.maxim_sort_ascend(pn_locs[:pn_npks], pn_npks)
        
        return pn_locs, pn_npks


    def maxim_sort_ascend(self, pn_x, n_size):
        """
        Sort array in ascending order (insertion sort algorithm)
        """
        for i in range(1, n_size):
            n_temp = pn_x[i]
            j = i
            while j > 0 and n_temp < pn_x[j-1]:
                pn_x[j] = pn_x[j-1]
                j -= 1
            pn_x[j] = n_temp
        return pn_x
    

    def maxim_sort_indices_descend(self, pn_x, pn_indx, n_size):
        """
        Sort indices according to descending order (insertion sort algorithm)
        """
        for i in range(1, n_size):
            n_temp = pn_indx[i]
            j = i
            while j > 0 and pn_x[n_temp] > pn_x[pn_indx[j-1]]:
                pn_indx[j] = pn_indx[j-1]
                j -= 1
            pn_indx[j] = n_temp
        return pn_indx
