Positive / Negative visits - formatting
================
2018-02-22

Positive visits
===============

``` r
positive_visits <- 
  "positive_visits.tsv" %>% 
  read_tsv(col_types = "dddi")

positive_visits %>% head()
```

    ## # A tibble: 6 x 4
    ##   patient_id visit_id age_at_hosp_disch_in_days label
    ##        <dbl>    <dbl>                     <dbl> <int>
    ## 1     948607 16240200                      8312     1
    ## 2    2441771 14113020                     23797     1
    ## 3    2107368 30097904                     19520     1
    ## 4    2107368 29150039                     19518     1
    ## 5    2530410 34329134                     11674     1
    ## 6    2208801 14099165                       284     1

Negative visits
===============

``` r
if (file.exists("negative_visits.tsv")) {
  
  negative_visits <- 
    "negative_visits.tsv" %>% 
    read_tsv(col_types = "dddi")
  
} else {
  
  # Consider all potential negative visits and remove the ones that
  # are actually in the positive visits set
  negative_visits <- 
    "negative_visits_potential.txt" %>% 
    read_tsv(col_types = "ddd") %>% 
    anti_join(positive_visits, by = "visit_id") %>% 
    mutate(label = 0)

  negative_visits %>% 
    write_tsv("negative_visits.tsv")
}

negative_visits %>% 
  head()
```

    ## # A tibble: 6 x 4
    ##   visit_id patient_id age_at_hosp_disch_in_days label
    ##      <dbl>      <dbl>                     <dbl> <int>
    ## 1  9816656    1658455                      9597     0
    ## 2 33326278     621568                     18842     0
    ## 3 14399440     630772                     25893     0
    ## 4 10207893    1939746                     11855     0
    ## 5 20767038    1788396                     16646     0
    ## 6 15763251    1031831                     11688     0

All visits - shuffled
=====================

``` r
if (file.exists("all_visits.tsv")) {

  all_visits <- 
    "all_visits.tsv" %>% 
    read_tsv(col_types = "dddi")
  
} else {
  
  # Concantenate positive / negative visits and shuffle
  all_visits <- 
    positive_visits %>% 
    bind_rows(negative_visits) %>% 
    sample_frac()
  
  all_visits %>% 
    write_tsv("all_visits.tsv")
  
}

all_visits %>% head()
```

    ## # A tibble: 6 x 4
    ##   patient_id visit_id age_at_hosp_disch_in_days label
    ##        <dbl>    <dbl>                     <dbl> <int>
    ## 1    2592595 22066761                     15105     0
    ## 2    1381867 19473834                     18412     0
    ## 3     193900 12137406                     29014     0
    ## 4     614266 32397014                     18226     1
    ## 5    1631767  9726343                     18300     0
    ## 6    2212597  7936742                     21472     0
