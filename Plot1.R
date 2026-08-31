library(ggplot2)

ggplot(df, aes(y = Metric, x = Estimate)) +
  geom_point(size = 4) +
  geom_errorbarh(
    aes(xmin = Lower, xmax = Upper),
    height = 0.15,
    linewidth = 0.8
  ) +
  geom_text(
    aes(label = sprintf("%.2f%%", Estimate)),
    hjust = -0.2,
    size = 4
  ) +
  labs(
    title = "Bootstrap Confidence Intervals for Retrieval Performance",
    x = "Accuracy (%)",
    y = NULL
  ) +
  xlim(0, 65) +
  theme_classic(base_size = 14) +
  theme(
    plot.title = element_text(
      hjust = 0.5,
      face = "bold"
    )
  )