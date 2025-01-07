# Libraries
# ===================================
library(readr)
library(dplyr)
library(ggplot2)
library(ggtext)

# Extract Data
# ===================================
# URL GitHub
url <- "https://raw.githubusercontent.com/guillemmaya92/Python/main/Data/Catalunya_CP.csv"

# Read CSV
df <- read_delim(url, delim = ";", locale = locale(encoding = "latin1"))

# Select relevant columns and filter data
df <- df %>%
  select(province, region, price) %>%
  filter(region %in% c("Barcelonès") & price < 2000) %>%
  filter(!is.na(price))

# Transform Data
# ===================================
# Calculate values
min_price <- 0
cheaper <- 800
median_price <- median(df$price, na.rm = TRUE)
max_price <- 2000
total_announcements <- nrow(df)

mid1 <- (cheaper + min_price) / 2
announcements1 <- nrow(df %>% filter(price > min_price & price <= cheaper))

mid2 <- (median_price + cheaper) / 2
announcements2 <- nrow(df %>% filter(price > cheaper & price <= median_price))

mid3 <- (max_price + median_price) / 2
announcements3 <- nrow(df %>% filter(price > median_price & price <= max_price))

# Add color column
df <- df %>%
  mutate(color = case_when(
    price < cheaper ~ "#ffc939",
    price < median_price ~ "#a8c2d2",
    TRUE ~ "#477794"
  ))

# Show data
print(head(df))

# Plot Data
# ===================================
df %>%
  ggplot(aes(x = price, fill = after_stat(case_when(
    x <= cheaper ~ "cheaper",
    x <= median_price ~ "median",
    TRUE ~ "expensive"
  )))) +
  geom_dots(
    smooth = smooth_bounded(adjust = 0.8), 
    side = "both", 
    color = NA,
    dotsize = 0.8,
    stackratio = 1.3
  ) +
  scale_x_continuous(
    limits = c(min_price, max_price),
    breaks = seq(min_price, max_price, by = 200),
    labels = scales::comma_format()
  ) +
  scale_y_continuous(breaks = NULL) +
  labs(
    title = 'Pisos ofertados en Idealista por menos de 2.000 euros',
    subtitle = "Anuncios en la comarca del Barcelonès",
    x = "Precio (€)",
    caption = paste0(
      "**Fuente**: Idealista<br>**Notas**: Cada bola representa un anuncio"
    )
  ) +
  scale_fill_manual(values = c(
    "cheaper" = "#ffc939",
    "median" = "#a8c2d2",
    "expensive" = "#477794"
  )) +
  annotate("text", 
           x = 0, 
           y = 0.05, 
           label = "Barcelonès", 
           size = 4, 
           color = "black", 
           fontface = "bold", 
           hjust = 0) +
  annotate("text", 
           x = 0, 
           y = -0.05, 
           label = paste("Total anuncios:", comma(total_announcements)), 
           size = 4, 
           color = "black", 
           fontface = "plain", 
           hjust = 0) +
  annotate(geom = "label", 
           x = mid1, 
           y = 0.8, 
           label = paste(comma(announcements1), "pisos"), 
           size = 4, 
           color = "black", 
           fontface = "plain",
           fill = "#a68221",
           alpha = 0.3,
           label.size = 0) +
  annotate(geom = "text", 
           x = mid1, 
           y = 0.73, 
           label = paste("Entre", min_price, "y", cheaper), 
           size = 3, 
           color = "#909090") +
  annotate(geom = "label", 
           x = mid2, 
           y = 0.8, 
           label = paste(comma(announcements2), "pisos"), 
           size = 4, 
           color = "black", 
           fontface = "plain",
           fill = "grey",
           alpha = 0.3,
           label.size = 0) +
  annotate(geom = "text", 
           x = mid2, 
           y = 0.73, 
           label = paste("Entre", cheaper, "y", median_price), 
           size = 3, 
           color = "#909090") +
  annotate(geom = "label", 
           x = mid3, 
           y = 0.8, 
           label = paste(comma(announcements3), "pisos"), 
           size = 4, 
           color = "black", 
           fontface = "plain",
           fill = "#477794",
           alpha = 0.3,
           label.size = 0) +
  annotate(geom = "text", 
           x = mid3, 
           y = 0.73, 
           label = paste("Entre", median_price, "y", max_price), 
           size = 3, 
           color = "#909090") +
  geom_hline(yintercept = 0, linetype = "solid", color = "grey", size = 0.5) +
  geom_vline(xintercept = cheaper, color = "#9c7a1f", linetype = "dotted", size = 0.25) +
  geom_vline(xintercept = median_price, color = "#477794", linetype = "dotted", size = 0.25) +
  theme_minimal() +
  theme(
    plot.title = element_text(face = "bold"),
    plot.subtitle = element_text(face = "plain"),
    axis.title.x = element_text(face = "bold"),
    panel.grid.major.x = element_blank(),
    panel.grid.minor.x = element_blank(),
    panel.grid.major.y = element_blank(),
    panel.grid.minor.y = element_blank(),
    legend.position = "none",
    axis.title.y = element_blank(),
    plot.caption = element_markdown(size = 10, hjust = 0)
  )

# Saving Plot
ggsave("C:/Users/guill/Downloads/SCRAPERIUM/grafico.jpeg", 
       plot = last_plot(), dpi = 300, width = 10, height = 6)
