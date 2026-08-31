library(tidyverse)

df <- tribble(
  ~Model, ~Top1, ~Top5, ~Top10,
  "PCA", 0.13, 1.19, 1.98,
  "Contrastive Transformer", 7.27, 33.03, 56.14,
  "Multi-task Transformer", 0.53, 2.51, 4.36
)

df_long <- df %>%
  pivot_longer(
    cols = c(Top1, Top5, Top10),
    names_to = "Metric",
    values_to = "Accuracy"
  )

ggplot(df_long,
       aes(x = Metric,
           y = Accuracy,
           fill = Model)) +
  geom_bar(
    stat = "identity",
    position = position_dodge(width = 0.8),
    width = 0.7
  ) +
  geom_text(
    aes(label = sprintf("%.1f", Accuracy)),
    position = position_dodge(width = 0.8),
    vjust = -0.4,
    size = 3.5
  ) +
  labs(
    title = "Teacher–Child Retrieval Performance Across Models",
    x = NULL,
    y = "Accuracy (%)",
    fill = "Model"
  ) +
  theme_minimal(base_size = 14) +
  theme(
    plot.title = element_text(
      hjust = 0.5,
      face = "bold"
    ),
    panel.grid.minor = element_blank()
  )


ggsave(
  "model_comparison.pdf",
  width = 7,
  height = 5
)