library('mice')

handleOutlier <- function(df,detect_methods,handle_methods){
	if(detect_methods not in ("boxplot","grubbs","dixon")){
		print("input detect_methods are "boxplot","grubbs" or "dixon".")
}
	if(detect_methods=="boxplot"){
		for(i in 1:dim(df)[2]){
			QL <- quantile(df[,i], probs = 0.25)
    			QU <- quantile(df[,i], probs = 0.75)
    			QU_QL <- QU-QL
			df[,i][which(df[,i] > QU + 1.5*QU_QL)]
			# 用离异常点最近的点替换
    			handle01 <- df[,i]
    			out_imp01 <- max(handle01[which(handle01 <= QU + 1.5*QU_QL)])
    			handle01[which(handle01 > QU + 1.5*QU_QL)] <- out_imp01
			#异常值变为空，转化为缺失值问题
			handle02 <- df[,i]
			handle02[which(handle02 > QU + 1.5*QU_QL)] <- NULL

}
}
	if(detect_methods=="grubbs"){
		for(i in 1:dim(df)[2]){
			if(grubbs.test(df[,i])$p-value < 0.05){
				print('df[,i]有异常值')
				outlier <- grubbs.test(df[,i])$alternative hypothesis
				df[,i][which(df[,i] == str_extract_all(outlier,"[0-9]+%")] <- NULL
}			
}




}
	if(detect_methods=="dixon"){
		for(i in 1:dim(df)[2]){
			if(dixon.test(df[,i],type=11)$p-value < 0.05){
				print('df[,i]有异常值')
				outlier <- dixon.test(df[,i])$alternative hypothesis
				df[,i][which(df[,i] == str_extract_all(outlier,"[0-9]+%")] <- NULL

}
}
}