library('mice')

missingValue <- function(df,methods_list){
	if(sum(is.na(df)>0)){
		print(md.pattern(df))
		temp <- mice(df)
		fit <- with(temp,lm(df$target~.))
		pooled <- pool(fit)
		completed_df <- complete(temp,1)
}
}