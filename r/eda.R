library(ggplot2)

df        <- read.csv('../data/full_nonrookie.csv')
df_rookie <- read.csv('../data/full_rookie.csv')

g <- ggplot(data=df_rookie[which(df_rookie$draftYear>0),]) +
      geom_point(mapping=aes(x=draftYear, y=totalValue, col=draftOverall))
g
