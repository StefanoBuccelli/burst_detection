# loading useful libraries
library(pracma)
library(e1071)
# library(sjemea)

# source our burst detection functions
code.dir <- "C:\\Users\\BuccelliLab\\Documents\\GitHub\\burst_detection\\Burst_detection_methods\\"
code.files = dir(code.dir, pattern = "[.R]")
for (file in code.files){
  source(file = file.path(code.dir,file))
}

# source our sjemea github detection functions
code_sjemea.dir <- "C:\\Users\\BuccelliLab\\Documents\\GitHub\\sjemea\\R\\"
code_sjemea.files = dir(code_sjemea.dir, pattern = "[.R]")
for (file in code_sjemea.files){
  source(file = file.path(code_sjemea.dir,file))
}

# Load spikes
df <- scan("C:\\Users\\BuccelliLab\\Documents\\GitHub\\burst_detection\\spikes_s.txt")

# run burst detections

# run CMA and write to csv file
results.CMA=CMA.method(df)
write.csv(results.CMA, file = "C:\\Users\\BuccelliLab\\Documents\\GitHub\\burst_detection\\result_CMA.csv")

# run hennig_method and write to csv file
results.hennig_method=hennig.method(df)
write.csv(results.hennig_method, file = "C:\\Users\\BuccelliLab\\Documents\\GitHub\\burst_detection\\result_hennig_method.csv")


# run PS method and write to csv file
results.PS_method=PS.method(df,-log(0.01))
write.csv(results.PS_method, file = "C:\\Users\\BuccelliLab\\Documents\\GitHub\\burst_detection\\result_PS_method.csv")

# run RS method and write to csv file
results.RS_method=RS.method(df,-log(0.01)) # Minimum surprise value	log(0.01)  4.6
write.csv(results.RS_method, file = "C:\\Users\\BuccelliLab\\Documents\\GitHub\\burst_detection\\result_RS_method.csv")

# source our burst detection functions
code.dir <- "C:\\Users\\BuccelliLab\\Documents\\GitHub\\burst_detection\\Burst_detection_methods\\"
code.files = dir(code.dir, pattern = "[.R]")
for (file in code.files){
  source(file = file.path(code.dir,file))
}

# run logisi.pasq.method and write to csv file
results.pasquale_method=logisi.pasq.method(df)
write.csv(results.pasquale_method, file = "C:\\Users\\BuccelliLab\\Documents\\GitHub\\burst_detection\\result_pasquale_method.csv")

# source our sjemea github detection functions
code_sjemea.dir <- "C:\\Users\\BuccelliLab\\Documents\\GitHub\\sjemea\\R\\"
code_sjemea.files = dir(code_sjemea.dir, pattern = "[.R]")
for (file in code_sjemea.files){
  source(file = file.path(code_sjemea.dir,file))
}

# run MI_method and write to csv file
results.MI_method=MI.method(df)
write.csv(results.MI_method, file = "C:\\Users\\BuccelliLab\\Documents\\GitHub\\burst_detection\\result_MI_method.csv")

# run HSMM.method and write to csv file
results.HSMM_method=HSMM.method(df,0.5)
write.csv(results.HSMM_method, file = "C:\\Users\\BuccelliLab\\Documents\\GitHub\\burst_detection\\result_HSMM_method.csv")

# not working...

# run RGS.method and write to csv file
# results.RGS_method=RGS.method(df)
# write.csv(results.RGS_method, file = "C:\\Users\\BuccelliLab\\Documents\\GitHub\\burst_detection\\result_RGS_method.csv")







