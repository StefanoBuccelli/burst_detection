# loading useful libraries
library(pracma)
library(e1071)
library(sjemea)

# source our functions
code.dir <- "C:\\Users\\BuccelliLab\\Documents\\GitHub\\burst_detection\\Burst_detection_methods"
code.files = dir(code.dir, pattern = "[.r]")
for (file in code.files){
  source(file = file.path(code.dir,file))
}


# Load spikes
df <- scan("C:\\Users\\BuccelliLab\\Documents\\GitHub\\burst_detection\\spikes_s.txt")
# run CMA and write to csv file
results.CMA=CMA.method(df)
write.csv(results.CMA, file = "C:\\Users\\BuccelliLab\\Documents\\GitHub\\burst_detection\\result_CMA.csv")

# run hennig_method and write to csv file
results.hennig_method=hennig.method(df)
write.csv(results.hennig_method, file = "C:\\Users\\BuccelliLab\\Documents\\GitHub\\burst_detection\\result_hennig_method.csv")


# run PS method and write to csv file
results.PS_method=PS.method(df)
write.csv(results.PS_method, file = "C:\\Users\\BuccelliLab\\Documents\\GitHub\\burst_detection\\result_PS_method.csv")

# run RS method and write to csv file
results.RS_method=RS.method(df,-log(0.01)) # Minimum surprise value	???log(0.01) ??? 4.6
write.csv(results.RS_method, file = "C:\\Users\\BuccelliLab\\Documents\\GitHub\\burst_detection\\result_RS_method.csv")


# not working...

# run logisi.pasq.method and write to csv file
results.pasquale_method=logisi.pasq.method(df)
write.csv(results.pasquale_method, file = "C:\\Users\\BuccelliLab\\Documents\\GitHub\\burst_detection\\result_pasquale_method.csv")

# run MI_method and write to csv file
results.MI_method=MI.method(df)
write.csv(results.MI_method, file = "C:\\Users\\BuccelliLab\\Documents\\GitHub\\burst_detection\\results_MI_method.csv")

# run RGS.method and write to csv file
results.RGS_method=RGS.method(df)
write.csv(results.RGS_method, file = "C:\\Users\\BuccelliLab\\Documents\\GitHub\\burst_detection\\result_RGS_method.csv")