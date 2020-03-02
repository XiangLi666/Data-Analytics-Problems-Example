#2-2
with tb1 as
(
select Year, Quarter, Month,
       (select sum(Booking_Revenue)
        from table_2_3 t2
        where t2.Year = t.Year and
			  t2.Quarter = t.Quarter and
              t2.Month <= t.Month
       ) as Current_Year_Quarter_To_Date_Booking_Revenue
from table_2_3 t
),
tb2 as
(
select Year, Quarter, Month, Current_Year_Quarter_To_Date_Booking_Revenue as Last_Year_Quarter_To_Date_Booking_Revenue
from tb1
),
tb3 as
(
select tb1.Year,tb1.Quarter,tb1.Month,tb1.Current_Year_Quarter_To_Date_Booking_Revenue,tb2.Last_Year_Quarter_To_Date_Booking_Revenue 
from tb1 
left join tb2
on tb1.Year=tb2.Year+1
and tb1.Quarter=tb2.Quarter
and tb1.Month>=tb2.Month
)
select Year,Quarter,Month,Current_Year_Quarter_To_Date_Booking_Revenue,max(Last_Year_Quarter_To_Date_Booking_Revenue) as Last_Year_Quarter_To_Date_Booking_Revenue
from tb3
group by 1,2,3,4;

#2-3
with tb1 as
(
select Quarter,sum(Booking_Revenue) as Bookings_2005
from table_2_3
where year=2005
group by Quarter
),
tb2 as
(
select Quarter,sum(Booking_Revenue) as Bookings_2006
from table_2_3
where year=2006
group by Quarter
),
tb3 as
(
select Quarter,sum(Booking_Revenue) as Bookings_2007
from table_2_3
where year=2007
group by Quarter
)
select tb1.Quarter,tb1.Bookings_2005,tb2.Bookings_2006,tb3.Bookings_2007
from tb3
left join tb2 on tb3.Quarter=tb2.Quarter
left join tb1 on tb2.Quarter=tb1.Quarter;

#3-3
with dt as
(
select str_to_date("20070101","%Y%m%d") as q_start,str_to_date("20070331","%Y%m%d") as q_end
union 
select str_to_date("20070401","%Y%m%d") as q_start,str_to_date("20070630","%Y%m%d") as q_end
union 
select str_to_date("20070701","%Y%m%d") as q_start,str_to_date("20070930","%Y%m%d") as q_end
union 
select str_to_date("20071001","%Y%m%d") as q_start,str_to_date("20071231","%Y%m%d") as q_end
),
raw_name_list as
(
select concat(year(dt.q_start),"-Q",quarter(dt.q_start)) as period,dt.*,tb1.* from
dt left join table_3_10 tb1
on dt.q_end>=tb1.start_date and dt.q_end<=tb1.termination_date or 
dt.q_end>=tb1.start_date and tb1.termination_date is null or 
dt.q_end>=tb1.start_date and quarter(dt.q_end)=quarter(tb1.termination_date)
order by 1,2
)
select period,
count(case when termination_date is null or termination_date>=q_end then 1 end) as total_employees_at_end_of_quarter,
count(case when termination_date is null and performance_level="high" or termination_date>=q_end and performance_level="high" then 1 end) as high_performers,
count(case when termination_date is null and performance_level="medium" or termination_date>=q_end and performance_level="medium" then 1 end) as medium_performers,
count(case when termination_date is null and performance_level="low" or termination_date>=q_end and performance_level="low" then 1 end) as low_performers,
count(case when termination_date is not null and termination_date<=q_end then 1 end) as total_attrition,
count(case when termination_date is not null and termination_date<=q_end and performance_level="low" then 1 end) as low_performer_attrition,
count(case when termination_date is not null and termination_date<=q_end and performance_level="medium" then 1 end) as medium_performer_attrition,
count(case when termination_date is not null and termination_date<=q_end and performance_level="high" then 1 end) as high_performer_attrition
from raw_name_list
group by 1
order by 1;

#3-5
with dt as
(
select str_to_date("20160105","%Y%m%d") as date_
union 
select str_to_date("20160112","%Y%m%d") as date_
union 
select str_to_date("20160119","%Y%m%d") as date_
union 
select str_to_date("20160126","%Y%m%d") as date_
)
select dt.date_, count(*)
from dt 
left join table_3_14
on date_ between start_date and end_date
group by 1;


with tb1 as
(
select customer_id, statuss, datee
from table111
where statuss='start'
),
tb2 as
(
select customer_id, statuss, datee
from table111
where statuss='cancel'
),
tb3 as
(
select tb1.customer_id, tb1.datee as start_date, tb2.datee as end_date
from tb1
left join tb2
on tb1.customer_id=tb2.customer_id
and tb2.datee>tb1.datee
)
select customer_id, start_date, min(end_date) as end_date
from tb3
group by 1,2
order by 1,2;
