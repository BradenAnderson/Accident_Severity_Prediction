# Accident_Severity_Prediction

## Features
`regionname` [region] - no change (4 levels)

`urbanicityname` [urbanicity] - no change (2 levels)

`body_typname` - tavin's problem child

`mod_yearname` - sklearn impute 

`vtrafwayname` [vtrafway] - 7 levels + combine nr/unk/runk

`vnum_lanname` [vnum_lan] - 8 + combine nr/unk/runk

`vsurcondname` [vsurcond] - hot encode?

`vtrafconname` [vtrafcon] - hot encode or combine into 4 levels per pg 159?

`typ_intname` [typ_int] - change to yes, no, unknown

`int_hwyname` [int_hwy] - no change (3 levels)

`weathername` - hien's problem child

`wkdy_imname` [day_week] - no change

`reljct1` - keep as is, combine nr/unk/runk

`lgtcon_imname` - keep as is, combine other/nr/unk/runk

`maxsev_imname` [max_sev] - no change

`alchl_imname` [alcohol] - combine not applicable/unk

`age_im` - remove ages 15 and below, bin 80+

`sex_imname` [sex] - combine nr/unk/runk

`trav_sp` - no change but possibly remove traveling speed below X mph?

`hour_binned` - no change

`speeding_status` - no change

`rest_use` - bin into none, minimal, full, unknown

`p_crash1` - keep levels as is but combine other/unknwn

`ads_pres` - keep levels as but combine np/unk

nr = not reported /
unk = unknown /
runk = reported as unknown

[Data dictionary](https://crashstats.nhtsa.dot.gov/Api/Public/ViewPublication/813236)

## Removed
`hour_imname` [hour] - unnecessary with `hour_binned`

`vspd_lim` - unnecessary with `speeding_status`

`makename` - too many levels

`wrk_zonename` - not a useful feature after EDA

`reljct2_imname` - substituted with `reljct1`
