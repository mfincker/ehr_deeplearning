# One Hot
Hershel Mehta  
February 11, 2018  



# Setup Data

## Setup connection to stride7



## Setup diagnosis master 


```r
dx_master <- tbl(con, "dx_master")
```


```r
dx_master %>% head()
```

```
## # Source:   lazy query [?? x 8]
## # Database: mysql 10.1.30-MariaDB
## #   [hershelm@shahlab-db1.stanford.edu:/stride7]
##   patient_id  visit_id   contact_date_time AGE_AT_CONTACT_IN_DAYS    code
##        <dbl>     <dbl>               <chr>                  <dbl>   <chr>
## 1          1   9922763 2011-08-14 00:00:00                4335.00  139894
## 2          1 103602593 2015-10-16 08:29:00                5859.35 1359949
## 3          1 103602593 2015-10-16 08:29:00                5859.35 1298797
## 4          1 103602593 2015-10-16 08:29:00                5859.35 1298910
## 5          1  23782772 2014-12-12 00:00:00                5551.00  139894
## 6          1 101019436 2015-10-16 00:00:00                5859.35 1359949
## # ... with 3 more variables: code_source <chr>, table_source <chr>,
## #   dx_hospital <chr>
```

## Setup procedure master 


```r
px_master <- tbl(con, "px_master")
```


```r
px_master %>% head()
```

```
## # Source:   lazy query [?? x 8]
## # Database: mysql 10.1.30-MariaDB
## #   [hershelm@shahlab-db1.stanford.edu:/stride7]
##   patient_id  visit_id contact_date age_at_contact_in_days  code
##        <dbl>     <dbl>        <chr>                  <dbl> <chr>
## 1          1 107726161   2015-02-05                   5606  4984
## 2          1  30620595   2015-10-16                   5859 88341
## 3          1  30620595   2015-10-16                   5859 88300
## 4          1  23782772   2015-02-06                   5607 99232
## 5          1 107563330   2015-10-16                   5859 20680
## 6          1 107726161   2015-02-05                   5606  6594
## # ... with 3 more variables: code_source <chr>, table_source <chr>,
## #   px_hospital <chr>
```

## Setup medication master


```r
rx_master <- tbl(con, "SHC_med_de")
```


```r
rx_master %>% head(100)
```

```
## # Source:   lazy query [?? x 37]
## # Database: mysql 10.1.30-MariaDB
## #   [hershelm@shahlab-db1.stanford.edu:/stride7]
##    patient_id   med_id visit_id          order_time age_at_order_in_days
##         <dbl>    <dbl>    <dbl>               <chr>                <dbl>
##  1      85070 11109869 26244792 2015-01-22 18:05:00             17310.80
##  2    2226281  2221610 10356976 2011-10-13 12:01:00             25134.50
##  3     493461  2889041  8184775 2011-04-01 05:55:00             28348.20
##  4    1911472  3352180  6443408 2010-04-09 19:37:00             19605.80
##  5    1797018  3316724 10837875 2011-12-04 14:10:00             17835.60
##  6    2024700  4105820 12434519 2011-12-31 22:49:00             30551.00
##  7    2292063 13927137 24280564 2014-10-20 21:02:00             19010.90
##  8    1904039  7624523 13077407 2012-05-14 09:06:00             14359.40
##  9    2527619  7820452 14000089 2012-05-31 15:07:00             11221.60
## 10    1406135 16755231 24689772 2015-01-14 10:58:00              7986.46
## # ... with more rows, and 32 more variables: start_time <chr>,
## #   age_at_start_in_days <dbl>, end_time <chr>, age_at_end_in_days <dbl>,
## #   medication_id <dbl>, description <chr>, amd_med_disp_name <chr>,
## #   order_class_c <dbl>, ordering_mode_c <dbl>, sig <chr>, quantity <chr>,
## #   refills <chr>, med_presc_prov_id <dbl>, med_route_c <dbl>,
## #   discon_time <chr>, age_at_discon_in_days <dbl>,
## #   chng_order_med_id <dbl>, hv_discr_freq_id <chr>, freq_name <chr>,
## #   freq_display_name <chr>, freq_type <chr>, number_of_times <dbl>,
## #   time_unit <chr>, prn_yn <chr>, freq_period <chr>,
## #   hv_discrete_dose <chr>, hv_dose_unit_c <dbl>, order_status_c <dbl>,
## #   min_discrete_dose <dbl>, max_discrete_dose <dbl>, dose_unit_c <dbl>,
## #   lastdose <chr>
```


# Diagnoses (ICD9)

## Get ICD9 Sample


```r
icd9_test <- 
  dx_master %>% 
  filter(
    #patient_ids from maevz,
    code_source == "ICD9CM",
    dx_hospital == "SHC"
  ) %>% 
  select(
    patient_id, age_in_days = AGE_AT_CONTACT_IN_DAYS, code
  ) %>% 
  mutate(
    n = 1
  ) %>% 
  head(100) %>% 
  collect()
```

```
## Warning in .local(conn, statement, ...): Decimal MySQL column 3 imported as
## numeric
```



## Sparse, One-Hot (each row is a single code)


```r
icd9_test %>% 
  distinct() %>% 
  mutate(
    code = str_c("ICD9_", code),
    id = rownames(.)
  ) %>% 
  spread(key = code, value = n, fill = 0)
```

```
## # A tibble: 99 x 73
##    patient_id age_in_days    id ICD9_233.0 ICD9_259.1 ICD9_274.0
##  *      <dbl>       <dbl> <chr>      <dbl>      <dbl>      <dbl>
##  1         12     8104.00    11          0          0          0
##  2         12     8104.00    17          0          0          0
##  3         12     8104.00    18          0          0          0
##  4         12     8104.00    20          0          0          0
##  5         12     8104.00     6          0          0          0
##  6         12     8104.00     7          0          0          0
##  7         12     8104.77    12          0          0          0
##  8         12     8104.77    19          0          0          0
##  9         12     8104.77    21          0          0          0
## 10         12     8111.00     1          0          0          0
## # ... with 89 more rows, and 67 more variables: ICD9_276.1 <dbl>,
## #   ICD9_276.8 <dbl>, ICD9_278.00 <dbl>, ICD9_311 <dbl>, ICD9_323.9 <dbl>,
## #   ICD9_333.99 <dbl>, ICD9_367.1 <dbl>, ICD9_382.9 <dbl>,
## #   ICD9_388.01 <dbl>, ICD9_401.1 <dbl>, ICD9_401.9 <dbl>,
## #   ICD9_424.1 <dbl>, ICD9_441.00 <dbl>, ICD9_470 <dbl>, ICD9_472.0 <dbl>,
## #   ICD9_473.0 <dbl>, ICD9_473.9 <dbl>, ICD9_478.19 <dbl>,
## #   ICD9_518.81 <dbl>, ICD9_530.11 <dbl>, ICD9_564.00 <dbl>,
## #   ICD9_565.0 <dbl>, ICD9_568.81 <dbl>, ICD9_569.0 <dbl>,
## #   ICD9_574.00 <dbl>, ICD9_574.10 <dbl>, ICD9_574.20 <dbl>,
## #   ICD9_575.12 <dbl>, ICD9_575.8 <dbl>, ICD9_593.9 <dbl>,
## #   ICD9_611.79 <dbl>, ICD9_620.8 <dbl>, ICD9_622.10 <dbl>,
## #   ICD9_623.8 <dbl>, ICD9_625.8 <dbl>, ICD9_625.9 <dbl>,
## #   ICD9_626.4 <dbl>, ICD9_628.9 <dbl>, ICD9_632 <dbl>, ICD9_633.10 <dbl>,
## #   ICD9_633.90 <dbl>, ICD9_633.91 <dbl>, ICD9_719.46 <dbl>,
## #   ICD9_724.00 <dbl>, ICD9_724.02 <dbl>, ICD9_724.2 <dbl>,
## #   ICD9_729.5 <dbl>, ICD9_780.6 <dbl>, ICD9_785.2 <dbl>,
## #   ICD9_786.05 <dbl>, ICD9_786.2 <dbl>, ICD9_787.2 <dbl>,
## #   ICD9_789.00 <dbl>, ICD9_799.0 <dbl>, ICD9_813.42 <dbl>,
## #   ICD9_922.4 <dbl>, ICD9_E885.2 <dbl>, ICD9_E917.0 <dbl>,
## #   ICD9_V06.5 <dbl>, ICD9_V22.2 <dbl>, ICD9_V67.00 <dbl>,
## #   ICD9_V70.0 <dbl>, ICD9_V72.31 <dbl>, ICD9_V72.83 <dbl>,
## #   ICD9_V72.84 <dbl>, ICD9_V76.12 <dbl>, ICD9_V84.01 <dbl>
```

## Sparse (each row is a single day)


```r
icd9_test %>% 
  distinct() %>% 
  spread(key = code, value = n, fill = 0)
```

```
## # A tibble: 60 x 72
##    patient_id age_in_days `233.0` `259.1` `274.0` `276.1` `276.8` `278.00`
##  *      <dbl>       <dbl>   <dbl>   <dbl>   <dbl>   <dbl>   <dbl>    <dbl>
##  1         12     8104.00       0       0       0       0       0        0
##  2         12     8104.77       0       0       0       0       0        0
##  3         12     8111.00       0       0       0       0       0        0
##  4         12     8116.00       0       0       0       0       0        0
##  5         12     8653.00       0       0       0       0       0        0
##  6         12     8655.00       0       0       0       0       0        0
##  7         12     8655.78       0       0       0       0       0        0
##  8         12     8660.00       0       0       0       0       0        0
##  9         12     8720.00       0       0       0       0       0        0
## 10         77    27583.00       0       0       0       0       0        0
## # ... with 50 more rows, and 64 more variables: `311` <dbl>,
## #   `323.9` <dbl>, `333.99` <dbl>, `367.1` <dbl>, `382.9` <dbl>,
## #   `388.01` <dbl>, `401.1` <dbl>, `401.9` <dbl>, `424.1` <dbl>,
## #   `441.00` <dbl>, `470` <dbl>, `472.0` <dbl>, `473.0` <dbl>,
## #   `473.9` <dbl>, `478.19` <dbl>, `518.81` <dbl>, `530.11` <dbl>,
## #   `564.00` <dbl>, `565.0` <dbl>, `568.81` <dbl>, `569.0` <dbl>,
## #   `574.00` <dbl>, `574.10` <dbl>, `574.20` <dbl>, `575.12` <dbl>,
## #   `575.8` <dbl>, `593.9` <dbl>, `611.79` <dbl>, `620.8` <dbl>,
## #   `622.10` <dbl>, `623.8` <dbl>, `625.8` <dbl>, `625.9` <dbl>,
## #   `626.4` <dbl>, `628.9` <dbl>, `632` <dbl>, `633.10` <dbl>,
## #   `633.90` <dbl>, `633.91` <dbl>, `719.46` <dbl>, `724.00` <dbl>,
## #   `724.02` <dbl>, `724.2` <dbl>, `729.5` <dbl>, `780.6` <dbl>,
## #   `785.2` <dbl>, `786.05` <dbl>, `786.2` <dbl>, `787.2` <dbl>,
## #   `789.00` <dbl>, `799.0` <dbl>, `813.42` <dbl>, `922.4` <dbl>,
## #   E885.2 <dbl>, E917.0 <dbl>, V06.5 <dbl>, V22.2 <dbl>, V67.00 <dbl>,
## #   V70.0 <dbl>, V72.31 <dbl>, V72.83 <dbl>, V72.84 <dbl>, V76.12 <dbl>,
## #   V84.01 <dbl>
```

## Original approach to one-hot encoding


```r
icd9_test %>% 
  select(-n) %>% 
  distinct() %>% 
  modelr::model_matrix( ~ . -1) %>% 
  arrange(patient_id, age_in_days)
```

```
## # A tibble: 99 x 72
##    patient_id age_in_days code233.0 code259.1 code274.0 code276.1
##         <dbl>       <dbl>     <dbl>     <dbl>     <dbl>     <dbl>
##  1         12     8104.00         0         0         0         0
##  2         12     8104.00         0         0         0         0
##  3         12     8104.00         0         0         0         0
##  4         12     8104.00         0         0         0         0
##  5         12     8104.00         0         0         0         0
##  6         12     8104.00         0         0         0         0
##  7         12     8104.77         0         0         0         0
##  8         12     8104.77         0         0         0         0
##  9         12     8104.77         0         0         0         0
## 10         12     8111.00         0         0         0         0
## # ... with 89 more rows, and 66 more variables: code276.8 <dbl>,
## #   code278.00 <dbl>, code311 <dbl>, code323.9 <dbl>, code333.99 <dbl>,
## #   code367.1 <dbl>, code382.9 <dbl>, code388.01 <dbl>, code401.1 <dbl>,
## #   code401.9 <dbl>, code424.1 <dbl>, code441.00 <dbl>, code470 <dbl>,
## #   code472.0 <dbl>, code473.0 <dbl>, code473.9 <dbl>, code478.19 <dbl>,
## #   code518.81 <dbl>, code530.11 <dbl>, code564.00 <dbl>, code565.0 <dbl>,
## #   code568.81 <dbl>, code569.0 <dbl>, code574.00 <dbl>, code574.10 <dbl>,
## #   code574.20 <dbl>, code575.12 <dbl>, code575.8 <dbl>, code593.9 <dbl>,
## #   code611.79 <dbl>, code620.8 <dbl>, code622.10 <dbl>, code623.8 <dbl>,
## #   code625.8 <dbl>, code625.9 <dbl>, code626.4 <dbl>, code628.9 <dbl>,
## #   code632 <dbl>, code633.10 <dbl>, code633.90 <dbl>, code633.91 <dbl>,
## #   code719.46 <dbl>, code724.00 <dbl>, code724.02 <dbl>, code724.2 <dbl>,
## #   code729.5 <dbl>, code780.6 <dbl>, code785.2 <dbl>, code786.05 <dbl>,
## #   code786.2 <dbl>, code787.2 <dbl>, code789.00 <dbl>, code799.0 <dbl>,
## #   code813.42 <dbl>, code922.4 <dbl>, codeE885.2 <dbl>, codeE917.0 <dbl>,
## #   codeV06.5 <dbl>, codeV22.2 <dbl>, codeV67.00 <dbl>, codeV70.0 <dbl>,
## #   codeV72.31 <dbl>, codeV72.83 <dbl>, codeV72.84 <dbl>,
## #   codeV76.12 <dbl>, codeV84.01 <dbl>
```

\pagebreak

# Procedures (CPT)


```r
cpt_test <- 
  px_master %>% 
  filter(
    #patient_ids from maevz,
    code_source == "CPT",
    px_hospital == "SHC"
  ) %>% 
  select(
    patient_id, age_in_days = age_at_contact_in_days, code
  ) %>% 
  mutate(
    n = 1
  ) %>% 
  head(100) %>% 
  collect()
```

```
## Warning in .local(conn, statement, ...): Decimal MySQL column 3 imported as
## numeric
```

\pagebreak

# Medications (RX_NORM)


```r
rx_test <- 
  rx_master %>%
  select(patient_id, age_in_days = age_at_order_in_days, code = med_id) %>% 
  mutate(
    n = 1
  ) %>% 
  head(100) %>% 
  collect()
```

```
## Warning in .local(conn, statement, ...): Decimal MySQL column 3 imported as
## numeric
```

\pagebreak

# Combine


```r
codes_test <- 
  icd9_test %>% 
  distinct() %>% 
  mutate(
    code = str_c("ICD9_", code)
  ) %>% 
  union(
    cpt_test %>% 
      distinct() %>% 
      mutate(
        code = str_c("CPT_", code)
      )
  ) %>% 
  union(
    rx_test %>% 
      distinct() %>% 
      mutate(
        code = str_c("RXNORM_", code)
      )
  )
```


```r
codes_test %>% 
  distinct(code)
```

```
## # A tibble: 234 x 1
##               code
##              <chr>
##  1   RXNORM_475161
##  2 RXNORM_16514458
##  3  RXNORM_5192254
##  4  RXNORM_9837354
##  5 RXNORM_29279169
##  6 RXNORM_26198119
##  7 RXNORM_14397530
##  8 RXNORM_33746443
##  9 RXNORM_12373124
## 10 RXNORM_31515028
## # ... with 224 more rows
```


```r
codes_test %>% 
  mutate(
    id = rownames(.)
  ) %>% 
  spread(key = code, value = n, fill = 0) %>% 
  arrange(patient_id, age_in_days)
```

```
## # A tibble: 296 x 237
##    patient_id age_in_days    id CPT_00840 CPT_01610 CPT_14000 CPT_19281
##         <dbl>       <dbl> <chr>     <dbl>     <dbl>     <dbl>     <dbl>
##  1          1        4335   195         0         0         0         0
##  2          1        5606   179         0         0         0         0
##  3          1        5607   193         0         0         0         0
##  4          1        5859   191         0         0         0         0
##  5          1        5859   192         0         0         0         0
##  6          1        5859   194         0         0         0         0
##  7          1        5859   196         0         0         0         0
##  8          1        5859   197         0         0         0         0
##  9          1        5859   198         0         0         0         0
## 10          5       30226   199         0         0         0         0
## # ... with 286 more rows, and 230 more variables: CPT_19301 <dbl>,
## #   CPT_19302 <dbl>, CPT_20680 <dbl>, CPT_21012 <dbl>, CPT_38525 <dbl>,
## #   CPT_38900 <dbl>, CPT_49320 <dbl>, CPT_70450 <dbl>, CPT_71010 <dbl>,
## #   CPT_72072 <dbl>, CPT_72100 <dbl>, CPT_72125 <dbl>, CPT_72128 <dbl>,
## #   CPT_72131 <dbl>, CPT_72170 <dbl>, CPT_73030 <dbl>, CPT_73501 <dbl>,
## #   CPT_73550 <dbl>, CPT_73560 <dbl>, CPT_74176 <dbl>, CPT_76098 <dbl>,
## #   CPT_76801 <dbl>, CPT_76805 <dbl>, CPT_76817 <dbl>, CPT_76830 <dbl>,
## #   CPT_76856 <dbl>, CPT_77059 <dbl>, CPT_78195 <dbl>, CPT_88300 <dbl>,
## #   CPT_88305 <dbl>, CPT_88307 <dbl>, CPT_88321 <dbl>, CPT_88341 <dbl>,
## #   CPT_88342 <dbl>, CPT_90471 <dbl>, CPT_90472 <dbl>, CPT_90691 <dbl>,
## #   CPT_90734 <dbl>, CPT_92004 <dbl>, CPT_92015 <dbl>, CPT_92310 <dbl>,
## #   CPT_93010 <dbl>, CPT_93016 <dbl>, CPT_93018 <dbl>, CPT_93350 <dbl>,
## #   CPT_99024 <dbl>, CPT_99202 <dbl>, CPT_99203 <dbl>, CPT_99205 <dbl>,
## #   CPT_99212 <dbl>, CPT_99213 <dbl>, CPT_99214 <dbl>, CPT_99232 <dbl>,
## #   CPT_99233 <dbl>, CPT_99241 <dbl>, CPT_99244 <dbl>, CPT_99245 <dbl>,
## #   CPT_99284 <dbl>, CPT_99285 <dbl>, CPT_99291 <dbl>, CPT_99401 <dbl>,
## #   ICD9_233.0 <dbl>, ICD9_259.1 <dbl>, ICD9_274.0 <dbl>,
## #   ICD9_276.1 <dbl>, ICD9_276.8 <dbl>, ICD9_278.00 <dbl>, ICD9_311 <dbl>,
## #   ICD9_323.9 <dbl>, ICD9_333.99 <dbl>, ICD9_367.1 <dbl>,
## #   ICD9_382.9 <dbl>, ICD9_388.01 <dbl>, ICD9_401.1 <dbl>,
## #   ICD9_401.9 <dbl>, ICD9_424.1 <dbl>, ICD9_441.00 <dbl>, ICD9_470 <dbl>,
## #   ICD9_472.0 <dbl>, ICD9_473.0 <dbl>, ICD9_473.9 <dbl>,
## #   ICD9_478.19 <dbl>, ICD9_518.81 <dbl>, ICD9_530.11 <dbl>,
## #   ICD9_564.00 <dbl>, ICD9_565.0 <dbl>, ICD9_568.81 <dbl>,
## #   ICD9_569.0 <dbl>, ICD9_574.00 <dbl>, ICD9_574.10 <dbl>,
## #   ICD9_574.20 <dbl>, ICD9_575.12 <dbl>, ICD9_575.8 <dbl>,
## #   ICD9_593.9 <dbl>, ICD9_611.79 <dbl>, ICD9_620.8 <dbl>,
## #   ICD9_622.10 <dbl>, ICD9_623.8 <dbl>, ICD9_625.8 <dbl>,
## #   ICD9_625.9 <dbl>, ...
```



