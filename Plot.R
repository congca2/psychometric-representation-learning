library(tidyverse)

df <- tribble(
  ~Clusters, ~Silhouette, ~Model,
  2, 0.595, "Multi-task Transformer",
  3, 0.436, "Multi-task Transformer",
  4, 0.378, "Multi-task Transformer",
  5, 0.344, "Multi-task Transformer",
  2, 0.076, "Transformer",
  3, 0.077, "Transformer",
  4, 0.078, "Transformer",
  5, 0.072, "Transformer",
  6, 0.080, "Transformer",
  7, 0.082, "Transformer",
  8, 0.085, "Transformer",
  9, 0.084, "Transformer",
  10, 0.083, "Transformer"
)

ggplot(df,
       aes(x = Clusters,
           y = Silhouette,
           color = Model,
           group = Model)) +
  geom_line(linewidth = 1) +
  geom_point(size = 3) +
  labs(
    title = "Silhouette Coefficients Across Cluster Solutions",
    x = "Number of Clusters",
    y = "Silhouette Coefficient",
    color = NULL
  ) +
  theme_minimal(base_size = 14) +
  theme(
    plot.title = element_text(
      hjust = 0.5,
      face = "bold"
    )
  )

geom_vline(
  xintercept = 2,
  linetype = "dashed"
)
geom_text(
  data = subset(df,
                (Model=="Multi-task Transformer" & Clusters==2) |
                  (Model=="Transformer" & Clusters==8)),
  aes(label = paste0("K=", Clusters))
)
